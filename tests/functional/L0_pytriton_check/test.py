#!/usr/bin/env python3
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
"""Test of pytriton check tool."""

import argparse
import logging
import signal
import sys
import time

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{TEST_CONTAINER_VERSION}-py3",
}


def verify_output(output):
    LOGGER.info("Results matches expected results")

    matches_strings = [
        "add: [[2.0], [2.0]]",
        "sub: [[0.0], [0.0]]",
        "nvidia-smi process finished with 0",
        "Checking OS version",
        'OS version: "Ubuntu 22.04.',
        "Checking installed packages",
        "nvidia-pytriton==",
        "Checking psutil stats",
        "Checking compiler and C libraries",
    ]

    negative_matches_strings = [
        "Error occurred in command:",
        "OS release file not found",
        "uname failed",
        "gcc failed",
        "Failed to get glibc version",
    ]

    for match_string in matches_strings:
        if match_string not in output:
            raise ValueError("Couldn't find expected result: %s", match_string)
        else:
            LOGGER.info("Found expected result: %s", match_string)

    for match_string in negative_matches_strings:
        if match_string in output:
            raise ValueError("Found unexpected result: %s", match_string)
        else:
            LOGGER.info("Didn't find unexpected result: %s", match_string)

    LOGGER.info("All checks passed")


def main():
    from pytriton.check.utils import (
        DEFAULT_LOG_FORMAT,
        ScriptThread,
        get_current_container_version,
        verify_docker_image_in_readme_same_as_tested,
    )

    parser = argparse.ArgumentParser(description="short_description")
    parser.add_argument("--timeout-s", required=False, default=300, type=float, help="Timeout for test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format=DEFAULT_LOG_FORMAT)
    logging.captureWarnings(True)

    docker_image_with_name = METADATA["image_name"].format(TEST_CONTAINER_VERSION=get_current_container_version())
    verify_docker_image_in_readme_same_as_tested("examples/vllm/README.md", docker_image_with_name)

    start_time = time.time()
    elapsed_s = 0
    wait_time_s = min(args.timeout_s, 1)

    check_cmd = ["pytriton", "check"]

    with ScriptThread(check_cmd, name="check") as check_thread:
        while check_thread.is_alive() and elapsed_s < args.timeout_s:
            elapsed_s = time.time() - start_time

        LOGGER.info("Interrupting server script process")
        if check_thread.process:
            check_thread.process.send_signal(signal.SIGINT)
            check_thread.join(timeout=wait_time_s)

    timeout = elapsed_s >= args.timeout_s and check_thread.is_alive()
    if timeout:
        LOGGER.error("Timeout occurred (timeout_s=%s)", args.timeout_s)
        sys.exit(-2)

    verify_output(check_thread.output)


if __name__ == "__main__":
    main()
