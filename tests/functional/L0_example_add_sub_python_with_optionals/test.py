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
"""Test of add_sub_python_with_optionals example"""
import argparse
import logging
import signal
import subprocess
import sys
import time

from tests.utils import (
    DEFAULT_LOG_FORMAT,
    ScriptThread,
    get_current_container_version,
    verify_docker_image_in_readme_same_as_tested,
)

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{TEST_CONTAINER_VERSION}-py3",
}

expected_results = [
    ["add: [[4.0], [4.0]]", "sub: [[2.0], [2.0]]"],
    ["add: [[6.0], [6.0]]", "sub: [[4.0], [4.0]]"],
    ["add: [[7.0], [7.0]]", "sub: [[5.0], [5.0]]"],
    ["add: [[9.0], [9.0]]", "sub: [[7.0], [7.0]]"],
]


def verify_client_output(client_output):
    res = client_output.split("Received inference responses")
    chunks = res[-1].split("------------------------")[:-1]

    if len(chunks) != len(expected_results):
        raise ValueError("Couldn't find expected result")

    for out_chunk, expecte_res in zip(chunks, expected_results):
        for expected in expecte_res:
            if expected not in out_chunk:
                raise ValueError("Couldn't find expected result")

    LOGGER.info("Results matches expected results")


def main():
    parser = argparse.ArgumentParser(description="short_description")
    parser.add_argument("--timeout-s", required=False, default=300, type=float, help="Timeout for test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format=DEFAULT_LOG_FORMAT)

    docker_image_with_name = METADATA["image_name"].format(TEST_CONTAINER_VERSION=get_current_container_version())
    verify_docker_image_in_readme_same_as_tested(
        "examples/add_sub_python_with_optional/README.md", docker_image_with_name
    )

    subprocess.run(["bash", "examples/add_sub_python_with_optional/install.sh"])

    start_time = time.time()
    elapsed_s = 0
    wait_time_s = min(args.timeout_s, 1)

    server_cmd = ["python", "examples/add_sub_python_with_optional/server.py"]
    client_cmd = ["python", "examples/add_sub_python_with_optional/client.py"]

    with ScriptThread(server_cmd, name="server") as server_thread:
        with ScriptThread(client_cmd, name="client") as client_thread:
            while server_thread.is_alive() and client_thread.is_alive() and elapsed_s < args.timeout_s:
                client_thread.join(timeout=wait_time_s)
                elapsed_s = time.time() - start_time

        LOGGER.info("Interrupting server script process")
        if server_thread.process:
            server_thread.process.send_signal(signal.SIGINT)

    if client_thread.returncode != 0:
        raise RuntimeError(f"Client returned {client_thread.returncode}")
    if server_thread.returncode not in [0, -2]:  # -2 is returned when process finished after receiving SIGINT signal
        raise RuntimeError(f"Server returned {server_thread.returncode}")

    timeout = elapsed_s >= args.timeout_s and client_thread.is_alive() and server_thread.is_alive()
    if timeout:
        LOGGER.error(f"Timeout occurred (timeout_s={args.timeout_s})")
        sys.exit(-2)

    verify_client_output(client_thread.output)


if __name__ == "__main__":
    main()
