#!/usr/bin/env python3
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
"""Test of perf_analyzer example"""

import argparse
import contextlib
import logging
import re
import signal
import sys
import time
from multiprocessing.util import DEFAULT_LOGGING_FORMAT

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{TEST_CONTAINER_VERSION}-py3",
}


def verify_client_output(client_output):
    expected_patterns = [
        "Inferences/Second vs. Client Average Batch Latency",
        r"Concurrency: 4, throughput: \d+.\d+ infer/sec, latency \d+ usec",
        r"Concurrency: 8, throughput: \d+.\d+ infer/sec, latency \d+ usec",
        r"Concurrency: 12, throughput: \d+.\d+ infer/sec, latency \d+ usec",
        r"Concurrency: 16, throughput: \d+.\d+ infer/sec, latency \d+ usec",
    ]

    for expected_pattern in expected_patterns:
        output_match = re.search(expected_pattern, client_output, re.MULTILINE)
        output_array = output_match.group(0) if output_match else None
        if not output_array:
            raise ValueError(f"Could not find {expected_pattern} in client output")
        else:
            LOGGER.info('Found "%s" in client output', expected_pattern)


def main():
    from pytriton.check.utils import (
        ScriptThread,
        get_current_container_version,
        search_warning_on_too_verbose_log_level,
        verify_docker_image_in_readme_same_as_tested,
    )
    from pytriton.client.utils import create_client_from_url, wait_for_model_ready

    parser = argparse.ArgumentParser(description="short_description")
    parser.add_argument("--timeout-s", required=False, default=300, type=float, help="Timeout for test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format=DEFAULT_LOGGING_FORMAT)

    docker_image_with_name = METADATA["image_name"].format(TEST_CONTAINER_VERSION=get_current_container_version())
    verify_docker_image_in_readme_same_as_tested("examples/perf_analyzer/README.md", docker_image_with_name)

    install_cmd = ["bash", "examples/perf_analyzer/install.sh"]
    with ScriptThread(install_cmd, name="install") as install_thread:
        install_thread.join()

    if install_thread.returncode != 0:
        raise RuntimeError(f"Install thread returned {install_thread.returncode}")

    start_time = time.time()
    elapsed_s = 0
    wait_time_s = min(args.timeout_s, 1)

    server_cmd = ["python", "examples/perf_analyzer/server.py"]
    client_cmd = ["bash", "examples/perf_analyzer/client.sh"]

    with ScriptThread(server_cmd, name="server") as server_thread:
        url = "http://127.0.0.1:8000"
        with contextlib.closing(create_client_from_url(url)) as client:
            wait_for_model_ready(client, "BART", timeout_s=args.timeout_s)

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
        LOGGER.error("Timeout occurred (timeout_s=%s)", args.timeout_s)
        sys.exit(-2)

    verify_client_output(client_thread.output)
    assert not search_warning_on_too_verbose_log_level(server_thread.output)


if __name__ == "__main__":
    main()
