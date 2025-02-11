#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
"""Test of simple_python_remote_mode example"""

import argparse
import logging
import signal
import sys
import time

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{TEST_CONTAINER_VERSION}-py3",
}


def verify_client_output(client_output):
    expected_matches = [
        "add: [[3.0], [5.0]]",
        "sub: [[-1.0], [-1.0]]",
        "mul: [[2.0], [6.0]]",
        "power: [[1.0], [8.0]]",
    ]
    for expected_match in expected_matches:
        if expected_match not in client_output:
            raise ValueError(f"Couldn't find expected result: {expected_match}")
    LOGGER.info("Results matches expected results")


def main():
    from pytriton.check.utils import (
        DEFAULT_LOG_FORMAT,
        ScriptThread,
        get_current_container_version,
        search_warning_on_too_verbose_log_level,
        verify_docker_image_in_readme_same_as_tested,
    )

    parser = argparse.ArgumentParser(description="short_description")
    parser.add_argument("--timeout-s", required=False, default=300, type=float, help="Timeout for test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format=DEFAULT_LOG_FORMAT)
    logging.captureWarnings(True)

    docker_image_with_name = METADATA["image_name"].format(TEST_CONTAINER_VERSION=get_current_container_version())
    verify_docker_image_in_readme_same_as_tested("examples/simple_python_remote_mode/README.md", docker_image_with_name)

    install_cmd = ["bash", "examples/simple_python_remote_mode/install.sh"]
    with ScriptThread(install_cmd, name="install") as install_thread:
        install_thread.join()

    if install_thread.returncode != 0:
        raise RuntimeError(f"Install thread returned {install_thread.returncode}")

    start_time = time.time()
    elapsed_s = 0
    wait_time_s = min(args.timeout_s, 1)

    server_cmd = ["python", "examples/simple_python_remote_mode/server_starting_triton.py"]
    server_remote_mul_cmd = ["python", "examples/simple_python_remote_mode/server_remote_mul.py"]
    server_remote_pow_cmd = ["python", "examples/simple_python_remote_mode/server_remote_power.py"]
    client_cmd = ["python", "examples/simple_python_remote_mode/client.py"]

    with ScriptThread(server_cmd, name="server") as server_thread:
        with ScriptThread(server_remote_mul_cmd, name="server_remote_mul") as server_remote_mul_thread:
            with ScriptThread(server_remote_pow_cmd, name="server_remote_pow") as server_remote_pow_thread:
                with ScriptThread(client_cmd, name="client") as client_thread:
                    while (
                        all([
                            server_thread.is_alive(),
                            server_remote_mul_thread.is_alive(),
                            server_remote_pow_thread.is_alive(),
                            client_thread.is_alive(),
                        ])
                        and elapsed_s < args.timeout_s
                    ):
                        client_thread.join(timeout=wait_time_s)
                        elapsed_s = time.time() - start_time

                LOGGER.info("Interrupting server script process")
                if server_remote_pow_thread.process:
                    server_remote_pow_thread.process.send_signal(signal.SIGINT)
            if server_remote_mul_thread.process:
                server_remote_mul_thread.process.send_signal(signal.SIGINT)
        if server_thread.process:
            server_thread.process.send_signal(signal.SIGINT)

    if client_thread.returncode != 0:
        raise RuntimeError(f"Client returned {client_thread.returncode}")
    if server_thread.returncode not in [0, -2]:  # -2 is returned when process finished after receiving SIGINT signal
        raise RuntimeError(f"Server returned {server_thread.returncode}")
    if server_remote_mul_thread.returncode not in [0, -2]:
        raise RuntimeError(f"Remote mul server returned {server_remote_mul_thread.returncode}")
    if server_remote_pow_thread.returncode not in [0, -2]:
        raise RuntimeError(f"Remote pow server returned {server_remote_pow_thread.returncode}")

    timeout = elapsed_s >= args.timeout_s and client_thread.is_alive() and server_thread.is_alive()
    if timeout:
        LOGGER.error("Timeout occurred (timeout_s=%s)", args.timeout_s)
        sys.exit(-2)

    verify_client_output(client_thread.output)
    assert not search_warning_on_too_verbose_log_level(server_thread.output)


if __name__ == "__main__":
    main()
