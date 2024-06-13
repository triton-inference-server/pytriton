# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""Environment checks."""

import logging
import os
import pathlib
import platform
import re
import sys

import psutil

from pytriton.check.utils import ScriptThread


def nvidia_smi(logger):
    """Run nvidia-smi.

    Args:
        logger: logger instance
    """
    logger.info("Running nvidia-smi")
    with ScriptThread(["nvidia-smi"], name="nvidia-smi") as nvidia_smi_thread:
        nvidia_smi_thread.join()
        logger.info(nvidia_smi_thread.output)
        if nvidia_smi_thread.returncode != 0:
            logger.error("nvidia-smi failed - possible cause: no GPU available or driver not installed")
            logger.error(
                "If running in WSL wit sudo, make sure to add nvidia-smi folder (e.g. /usr/lib/wsl/lib) to sudoers file!"
            )


def get_platform_info(logger):
    """Get platform information (OS, python, etc.).

    Args:
        logger: logger instance
    """
    logger.info("Checking OS version")
    logger.info("Script is running in docker:" + str(pathlib.Path("/.dockerenv").exists()))

    os_release_path = pathlib.Path("/etc/os-release")
    if os_release_path.exists():
        with os_release_path.open() as f:
            os_release = f.read()
            logger.info("OS release")
            logger.info(os_release)
            for line in os_release.split("\n"):
                if "PRETTY_NAME" in line:
                    os_version = line.split("=")[1].strip()
                    logger.info(f"OS version: {os_version}")
    else:
        logger.warning("OS release file not found (not available on some systems")

    logger.info("Get platform info")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"System: {platform.system()}")
    logger.info(f"Release: {platform.release()}")
    logger.info(f"Version: {platform.version()}")
    logger.info(f"Machine: {platform.machine()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Python implementation: {platform.python_implementation()}")
    logger.info(f"Python compiler: {platform.python_compiler()}")
    logger.info(f"Python build: {platform.python_build()}")
    logger.info(f"libc_ver: {platform.libc_ver()}")


def check_psutil_stats(logger):
    """Check psutil stats.

    Args:
        logger: logger instance
    """
    logger.info("Checking psutil stats")
    logger.info("Memory stats")
    logger.info(psutil.virtual_memory())
    logger.info("Swap stats")
    logger.info(psutil.swap_memory())
    logger.info("Disk stats")
    logger.info(psutil.disk_usage("/"))
    logger.info("Disk io countwers")
    logger.info(psutil.disk_io_counters())
    logger.info("CPU stats")
    logger.info(psutil.cpu_times())
    logger.info("Network stats")
    logger.info(psutil.net_io_counters())


def get_listening_processes(logger):
    """Get listening processes.

    Args:
        logger: logger instance
    """
    logger.info("Listening processes")
    processes = {proc.pid: proc.name for proc in psutil.process_iter(["pid", "name"])}
    connections = psutil.net_connections()
    listening_sockets = [conn for conn in connections if conn.status == "LISTEN"]

    for listening_socket in listening_sockets:
        process_name = None
        if listening_socket.pid is not None and listening_socket.pid in processes:
            process_name = processes[listening_socket.pid]
        logger.info(
            f"Process ID: {listening_socket.pid}, Name: {process_name}, Local Address: {listening_socket.laddr}, Remote Address: {listening_socket.raddr}, Status: {listening_socket.status}"
        )


def installed_packages(logger):
    """Get installed packages.

    Args:
        logger: logger instance
    """
    logger.info("Checking installed packages")
    import importlib_metadata

    packages = importlib_metadata.distributions()

    installed_pkg = sorted([f"{package.metadata['Name']}=={package.version} ({package._path})" for package in packages])
    installed_pkg_str = "\n[\n\t" + ",\n\t".join(installed_pkg) + "\n]"
    logger.info(installed_pkg_str)


def check_compiler_and_clib(logger):
    """Check compiler and C libraries.

    Args:
        logger: logger instance
    """
    logger.info("Checking compiler and C libraries")
    with ScriptThread(["gcc", "--version"], name="gcc_version") as gcc_version_thread:
        gcc_version_thread.join()
        logger.info("GCC version:")
        logger.info(gcc_version_thread.output)
        if gcc_version_thread.returncode != 0:
            logger.error("gcc failed")

    logger.info("Python version:")
    logger.info(sys.version)

    try:
        logger.info(os.confstr("CS_GNU_LIBC_VERSION"))
    except AttributeError as e:
        logger.error(f"Failed to get glibc version {e}")


def log_env_variables(logger):
    """Log environment variables.

    Args:
        logger: logger instance
    """
    logger.info("Environment variables")

    env_vars = os.environ.items()
    blacklist_patterns = [
        r".*token.*",
        r".*secret.*",
        r".*key.*",
        r".*password.*",
    ]

    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in blacklist_patterns]
    filtered_env_vars = [
        f"{key}={value}"
        for key, value in env_vars
        if not any(pattern.search(key) or pattern.search(value) for pattern in patterns)
    ]

    env_vars_str = "\n".join(filtered_env_vars)
    logger.info(env_vars_str)


def env_checks(logger: logging.Logger):
    """Run all environment checks.

    Args:
        logger: logger instance
    """
    logger.info("Running all environment checks")
    get_platform_info(logger)
    nvidia_smi(logger)
    installed_packages(logger)
    check_psutil_stats(logger)
    get_listening_processes(logger)
    check_compiler_and_clib(logger)
    log_env_variables(logger)
