# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils."""

import contextlib
import fcntl
import logging
import os
import pathlib
import re
import select
import socket
import subprocess
import threading
import typing

LOGGER = logging.getLogger(__name__)
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)8s - %(process)8d - %(threadName)s - %(name)s: %(message)s"


def _read_outputs(_process, _logger, _outputs):
    # Set stdout and stderr file descriptors to non-blocking mode
    try:
        fcntl.fcntl(_process.stdout, fcntl.F_SETFL, os.O_NONBLOCK)
        fcntl.fcntl(_process.stderr, fcntl.F_SETFL, os.O_NONBLOCK)
    except ValueError:  # when selecting on closed files
        return

    buffers = {_process.stdout: "", _process.stderr: ""}
    rds = [_process.stdout, _process.stderr]
    while rds:
        try:
            readable, _, _ = select.select(rds, [], [], 1)
        except ValueError:  # when selecting on closed files
            break

        for rd in readable:
            try:
                data = os.read(rd.fileno(), 4096)
                if not data:
                    rds.remove(rd)
                    continue

                decoded_data = data.decode("utf-8")
                buffers[rd] += decoded_data
                lines = buffers[rd].splitlines(keepends=True)

                if buffers[rd].endswith("\n"):
                    complete_lines = lines
                    buffers[rd] = ""
                else:
                    complete_lines = lines[:-1]
                    buffers[rd] = lines[-1]

                for line in complete_lines:
                    line = line.rstrip()
                    _logger.info(line)
                    _outputs.append(line)
            except OSError:  # Reading from an empty non-blocking file
                pass


class ScriptThread(threading.Thread):
    """A class that runs external script in a separate thread."""

    def __init__(self, cmd, workdir=None, group=None, target=None, name=None, args=(), kwargs=None) -> None:
        """Initializes the ScriptThread object."""
        super().__init__(group, target, name, args, kwargs, daemon=True)
        self.cmd = cmd
        self.workdir = workdir
        self._process_spawned_or_spawn_error_flag = None
        self.active = False
        self._process = None
        self.returncode = None
        self._output = []
        self._logger = logging.getLogger(self.name)

    def __enter__(self):
        """Starts the script thread."""
        self.start(threading.Event())
        self._process_spawned_or_spawn_error_flag.wait()
        return self

    def __exit__(self, *args):
        """Stops the script thread and waits for it to join."""
        self.stop()
        self.join()
        self._process_spawned_or_spawn_error_flag = None

    def start(self, flag: typing.Optional[threading.Event] = None) -> None:
        """Starts the script thread."""
        if flag is None:
            flag = threading.Event()
        self._logger.info(f"Starting {self.name} script with \"{' '.join(self.cmd)}\" cmd")
        self._process_spawned_or_spawn_error_flag = flag
        super().start()

    def stop(self):
        """Sets the active flag to False to stop the script thread."""
        self._logger.info(f"Stopping {self.name} script")
        self.active = False

    def run(self):
        """Runs the script in a separate process."""
        import psutil

        self.returncode = None
        self._output = []
        self._process = None

        os.environ.setdefault("PYTHONUNBUFFERED", "1")  # to not buffer logs
        try:
            with psutil.Popen(
                self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, cwd=self.workdir
            ) as process:
                self._process = process
                self.active = True
                if self._process_spawned_or_spawn_error_flag:
                    self._process_spawned_or_spawn_error_flag.set()
                while self.active and process.poll() is None and process.returncode is None:
                    try:
                        _read_outputs(process, self._logger, self._output)
                    except KeyboardInterrupt:
                        self.stop()

        finally:
            if self._process_spawned_or_spawn_error_flag:
                self._process_spawned_or_spawn_error_flag.set()
            if self.process:
                while self.process.poll() is None:
                    _read_outputs(self.process, self._logger, self._output)
                _read_outputs(self.process, self._logger, self._output)
                self.returncode = process.wait()  # pytype: disable=name-error
                self._logger.info(f"{self.name} process finished with {self.returncode}")

            self.active = False
            self._process = None

    @property
    def output(self):
        """Return process stream output."""
        return "\n".join(self._output)

    @property
    def process(self):
        """Return process object."""
        return self._process


def find_free_port() -> int:
    """Finds a free port on the local machine."""
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class ProcessMonitoring:
    """A class that dumps the state of a process and its children.

    This class uses the py-spy tool to dump the stack trace of a process and its
    children recursively. It also dumps the process information such as the parent
    and the command line. It allows registering custom monitors that can perform
    additional actions on the process.

    Attributes:
        _logger (logging.Logger): The logger object to write messages.
        _process (psutil.Process): The process object to monitor.
        _children_processes (list[psutil.Process]): The list of child processes to monitor.
        _log (logging.Logger.method): The logging method to use for messages.
        _remove_color (bool): Whether to remove ANSI escape sequences from the output.
        _ansi_escape (re.Pattern): The regular expression object to match ANSI escape sequences.
        _custom_monitors (list[typing.Callable[[int], None]]): The list of custom monitor functions to execute on each dump cycle.
    """

    def __init__(
        self,
        pid: int,
        logger: typing.Optional[logging.Logger] = None,
        loglevel: int = logging.INFO,
        remove_color: bool = False,
    ):
        """Initializes the ProcessMonitoring object.

        Args:
            pid (int): The process ID of the process to monitor.
            logger (typing.Optional[logging.Logger], optional): The logger object to write messages. Defaults to None.
            loglevel (int, optional): The logging level to use for messages. Defaults to logging.INFO.
            remove_color (bool, optional): Whether to remove ANSI escape sequences from the output. Defaults to False.
        """
        import re

        import psutil

        self._logger = logger or logging.getLogger("monitoring")
        self._process = psutil.Process(pid)
        self._children_processes = list(self._process.children(recursive=True))
        self._log = {
            logging.DEBUG: self._logger.debug,
            logging.INFO: self._logger.info,
            logging.WARNING: self._logger.warning,
            logging.ERROR: self._logger.error,
        }[loglevel]
        self._log(f"Initial list of children processes: {self._children_processes}")
        self._remove_color = remove_color
        pattern = r"\x1b\[.*?m"
        self._ansi_escape = re.compile(pattern)
        self._custom_monitors = []

    def register_custom_monitor(self, custom_monitor: typing.Callable[[int], None]) -> None:
        """Registers a custom monitor for the process.

        This method adds a custom monitor function to the list of monitors that are
        executed on each dump cycle. A custom monitor function should take an integer
        as an argument (the process ID) and return None.

        Args:
            custom_monitor (typing.Callable[[int], None]): The custom monitor function to register.
        """
        self._custom_monitors.append(custom_monitor)

    def dump_state(self) -> None:
        """Dumps the state of the process and its children.

        This method calls the _dump_processes_stacktrace and _dump_child_processes
        methods to dump the stack trace and the process information of the process
        and its children recursively.
        """
        self._dump_processes_stacktrace()
        self._dump_child_processes()

    def _dump_processes_stacktrace(self):
        import psutil
        import sh

        self._log("==== Dump process stacktrace")
        pyspy_cmd = sh.Command("py-spy")

        for process in [self._process] + self.children:
            try:
                result = pyspy_cmd("dump", "-ll", "--nonblocking", "-p", str(process.pid))
                if self._remove_color:
                    result = self._ansi_escape.sub("", str(result))
                self._log(f"Dump stack trace for process (pid={process.pid}) with cmd {process.cmdline()}")
                for custom_monitor in self._custom_monitors:
                    custom_monitor(process.pid)
                self._log(result)
            except psutil.NoSuchProcess as e:
                self._log(f"Error during handling process: {e}")
            except sh.ErrorReturnCode_1 as e:
                self._log(f"Error during calling py-spy process: {e}")

    def _dump_child_processes(self):
        import psutil

        self._log("==== Dump process info (with its children)")
        for process in [self._process] + self.children:
            try:
                self._log(f"{process} parent={process.parent()} ")
            except psutil.NoSuchProcess:
                self._log(f"{process} is missing in process table")

    @property
    def children(self):
        """Returns the list of child processes to monitor.

        This property returns the list of child processes to monitor, and updates it
        with any new children that are created by the process.

        Returns:
            list[psutil.Process]: The list of child processes to monitor.
        """
        import psutil

        try:
            children = list(self._process.children(recursive=True))
            self._children_processes = list(set(self._children_processes + children))
        except psutil.NoSuchProcess:
            pass
        return self._children_processes


def get_current_container_version():
    """Returns the version of the current container."""
    container_version = os.environ.get("NVIDIA_PYTORCH_VERSION") or os.environ.get("NVIDIA_TENSORFLOW_VERSION")
    if container_version and "-" in container_version:
        container_version = container_version.split("-")[0]  # TF version has format <year_month_version>-<tf_version>
    return container_version


def verify_docker_image_in_readme_same_as_tested(readme_path, image_name_with_version):
    """Verify that the docker image is the same as described in the readme file."""
    image_name, _image_version = image_name_with_version.split(":")
    framework_name = image_name.split("/")[-1]
    readme_payload = pathlib.Path(readme_path).read_text()
    match_iterator = re.finditer(
        rf"(?P<container_registry>[\w/.\-:]+)/{framework_name}:(?P<image_version_with_python_version>[\w.-]+)",
        readme_payload,
    )
    for entry in match_iterator:
        assert entry.group() == image_name_with_version, f"{entry.group()} != {image_name_with_version}"


def search_warning_on_too_verbose_log_level(logs: str):
    """Search warnings."""
    pattern = r"Triton Inference Server is running with enabled verbose logs.*It may affect inference performance."
    return re.search(pattern, logs)


class ProcessMonitoringThread:
    """A class that creates a thread to monitor a process.

    This class uses the ProcessMonitoring class to dump the state of a process
    and its children periodically. It also allows registering custom monitors
    that can perform additional actions on the process.

    Attributes:
        _monitoring (ProcessMonitoring): The ProcessMonitoring object that handles the dumping logic.
        _stop_event (threading.Event): The event object that signals the thread to stop its loop.
        _thread (threading.Thread): The thread object that runs the _run method in a loop.
        _interval (float): The interval in seconds between each dump cycle.
    """

    def __init__(self, monitoring: ProcessMonitoring, interval: float = 60):
        """Initializes the ProcessMonitoringThread object.

        Args:
            monitoring (ProcessMonitoring): The ProcessMonitoring object that handles the dumping logic.
            interval (float, optional): The interval in seconds between each dump cycle. Defaults to 60.
        """
        self._monitoring = monitoring
        self._interval = interval

    def start(self) -> None:
        """Starts the monitoring thread.

        This method creates a new thread that runs the _run method in a loop until
        the stop method is called or an exception occurs. It also sets the stop event
        object that can be used to signal the thread to stop gracefully.
        """
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stops the monitoring thread.

        This method sets the stop event object that signals the thread to stop its loop.
        It also waits for the thread to join before returning.
        """
        self._stop_event.set()
        self._thread.join()

    def __enter__(self):
        """Enters the context manager for the monitoring thread."""
        self.start()
        return self

    def __exit__(self, *args):
        """Exits the context manager for the monitoring thread."""
        self.stop()

    def _run(self):
        logging.info("Monitoring process")
        self._monitoring.dump_state()
        while not self._stop_event.wait(self._interval):
            logging.info("Monitoring process")
            self._monitoring.dump_state()


class TestMonitoringContext:
    """A context manager that monitors test processes.

    This context manager creates threads to monitor the test processes and dumps
    their state periodically. It can extend argparse args with additional arguments.
    It supports splitting log into different files. The standard output log can have one level
    and the file log can have another level. It uses log rotation.
    """

    @staticmethod
    def extend_args(parser):
        """Extends argparse args with additional arguments."""
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Provide verbose logs",
        )
        parser.add_argument(
            "--log-path",
            type=str,
            default=None,
            help="Provide the path of external log for rotation",
        )
        parser.add_argument(
            "--compress-logs",
            action="store_true",
            help="Enable logs compression",
        )
        parser.add_argument(
            "--maximum-log-file",
            type=int,
            default=10 * 1024 * 1024,
            help="Maximum logfile size before rotation is started",
            required=False,
        )
        parser.add_argument(
            "--enable-fault-handler",
            action="store_true",
            help="Enable faulthandler",
        )
        parser.add_argument(
            "--faulthandler-interval",
            type=float,
            default=None,
            help="Enable faulthandler after specified number of seconds with repeat",
            required=False,
        )
        parser.add_argument(
            "--process-monitoring-interval",
            type=float,
            default=None,
            help="Enable process monitoring after specified number of seconds with repeat",
            required=False,
        )

    def __init__(self, args):
        """Initializes the TestMonitoringContext object.

        Args:
            args (argparse.Namespace): The argparse args object to extend with additional arguments.
        """
        self._args = args

    def __enter__(self):
        """Enters the context manager for the test monitoring."""
        import faulthandler
        import logging.handlers

        args = self._args
        self._loglevel = log_level = logging.DEBUG if args.verbose else logging.INFO
        logging.basicConfig(level=logging.DEBUG, format=DEFAULT_LOG_FORMAT)
        logger = logging.getLogger()

        if args.log_path is not None:
            # Create a rotating file handler for the file output logger
            # The file name is based on the log path argument, the maximum size is 10 MB, and the maximum number of files is 500
            file_handler = logging.handlers.RotatingFileHandler(
                args.log_path, maxBytes=args.maximum_log_file, backupCount=500
            )
            file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
            file_handler.setLevel(logging.DEBUG)
            if args.compress_logs:
                file_handler.namer = lambda name: name + ".gz"

                def gzip_rotation(source, dest):
                    import gzip
                    import os

                    with open(source, "rb") as f_in:
                        with gzip.open(dest, "wb") as f_out:
                            f_out.writelines(f_in)
                    os.remove(source)

                file_handler.rotator = gzip_rotation

            # Add the file handler to the default logger
            logger.addHandler(file_handler)
            # Get the stream handler that was created by basicConfig

            # Get the stream handler that was created by basicConfig
            stream_handler = logger.handlers[0]
            # Set the stream handler's level to match the log level argument
            stream_handler.setLevel(log_level)

            if args.enable_fault_handler:
                faulthandler.enable()

            if args.faulthandler_interval is not None:
                faulthandler.dump_traceback_later(args.faulthandler_interval, repeat=True, exit=False)

            custom_monitors = []

            import os

            import psutil

            def monitor_ram_usage(pid=None):
                if pid is None:
                    pid = os.getpid()

                process = psutil.Process(pid)
                logger.debug(f"MONITOR RAM USAGE ({pid}): {process.memory_info()}")

            custom_monitors.append(monitor_ram_usage)

            def monitor_file_descriptors(pid=None):
                if pid is None:
                    pid = os.getpid()

                process = psutil.Process(pid)
                logger.debug(f"MONITOR FILE DESCRIPTORS ({pid}): {process.num_fds()}")

            custom_monitors.append(monitor_file_descriptors)

            def monitor_cpu_usage(pid=None):
                if pid is None:
                    pid = os.getpid()

                process = psutil.Process(pid)
                logger.debug(f"MONITOR CPU USAGE ({pid}): {process.cpu_percent()}")

            custom_monitors.append(monitor_cpu_usage)

            def monitor_threads(pid=None):
                if pid is None:
                    pid = os.getpid()

                process = psutil.Process(pid)
                logger.debug(f"MONITOR THREADS ({pid}): {process.num_threads()}")

            custom_monitors.append(monitor_threads)

            def monitor_process_dict(pid=None):
                if pid is None:
                    pid = os.getpid()

                process = psutil.Process(pid)
                logger.debug(f"MONITOR PROCESS DICT ({pid}): {process.as_dict()}")

            custom_monitors.append(monitor_process_dict)
        if args.process_monitoring_interval is not None:
            monitoring = ProcessMonitoring(os.getpid(), logger, loglevel=logging.DEBUG, remove_color=True)
            for monitor in custom_monitors:
                monitoring.register_custom_monitor(monitor)

            self._monitor = ProcessMonitoringThread(monitoring, interval=args.process_monitoring_interval)
            self._monitor.start()
        return self

    def __exit__(self, *args):
        """Stops the monitor thread."""
        if hasattr(self, "_monitor"):
            self._monitor.stop()
            self._monitor = None
