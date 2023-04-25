# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
    def __init__(self, cmd, workdir=None, group=None, target=None, name=None, args=(), kwargs=None) -> None:
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
        self.start(threading.Event())
        self._process_spawned_or_spawn_error_flag.wait()
        return self

    def __exit__(self, *args):
        self.stop()
        self.join()
        self._process_spawned_or_spawn_error_flag = None

    def start(self, flag: threading.Event) -> None:
        self._logger.info(f"Starting {self.name} script with \"{' '.join(self.cmd)}\" cmd")
        self._process_spawned_or_spawn_error_flag = flag
        super().start()

    def stop(self):
        self._logger.info(f"Stopping {self.name} script")
        self.active = False

    def run(self):
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
        return "\n".join(self._output)

    @property
    def process(self):
        return self._process


def find_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class ProcessMonitoring:
    def __init__(self, pid: int):
        import psutil

        self._logger = logging.getLogger("monitoring")

        self._process = psutil.Process(pid)
        self._children_processes = list(self._process.children(recursive=True))
        self._logger.info(f"Initial list of children processes: {self._children_processes}")

    def dump_state(self):
        self._dump_processes_stacktrace()
        self._dump_child_processes()

    def _dump_processes_stacktrace(self):
        import psutil
        import sh

        self._logger.info("==== Dump process stacktrace")
        pyspy_cmd = sh.Command("py-spy")

        for process in [self._process] + self.children:
            try:
                self._logger.info(f"Dump stack trace for process (pid={process.pid}) with cmd {process.cmdline()}")
                result = pyspy_cmd("dump", "-ll", "--nonblocking", "-p", str(process.pid))
                self._logger.info(result)
            except psutil.NoSuchProcess as e:
                self._logger.info(f"Error during handling process: {e}")
            except sh.ErrorReturnCode_1 as e:
                self._logger.info(f"Error during calling py-spy process: {e}")

    def _dump_child_processes(self):
        import psutil

        self._logger.info("==== Dump process info (with its children)")
        for process in [self._process] + self.children:

            try:
                self._logger.info(f"{process} parent={process.parent()} ")
            except psutil.NoSuchProcess:
                self._logger.info(f"{process} is missing in process table")

    @property
    def children(self):
        import psutil

        try:
            children = list(self._process.children(recursive=True))
            self._children_processes = list(set(self._children_processes + children))
        except psutil.NoSuchProcess:
            pass
        return self._children_processes


def get_current_container_version():
    container_version = os.environ.get("NVIDIA_PYTORCH_VERSION") or os.environ.get("NVIDIA_TENSORFLOW_VERSION")
    if container_version and "-" in container_version:
        container_version = container_version.split("-")[0]  # TF version has format <year_month_version>-<tf_version>
    return container_version


def verify_docker_image_in_readme_same_as_tested(readme_path, image_name_with_version):
    image_name, image_version = image_name_with_version.split(":")
    framework_name = image_name.split("/")[-1]
    readme_payload = pathlib.Path(readme_path).read_text()
    match_iterator = re.finditer(
        rf"(?P<container_registry>[\w/.\-:]+)/{framework_name}:(?P<image_version_with_python_version>[\w.-]+)",
        readme_payload,
    )
    for entry in match_iterator:
        assert entry.group() == image_name_with_version, f"{entry.group()} != {image_name_with_version}"
