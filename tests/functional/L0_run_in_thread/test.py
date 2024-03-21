#!/usr/bin/env python3
# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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
"""Tests which:
- starting server
- wait for server to be ready
- sends SEGFAULT signal to server script
- waits till all children processes of server script finishes
"""

import argparse
import logging
import signal
import sys
import time

from tests.utils import (
    DEFAULT_LOG_FORMAT,
    ProcessMonitoring,  # pytype: disable=import-error
    ScriptThread,
    find_free_port,
)

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])

METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{TEST_CONTAINER_VERSION}-py3",
    "platforms": ["amd64", "arm64"],
}


def verify_client_output(client_output):
    output1_match = "Signal handled and triton server properly stopped" in client_output
    if not output1_match:
        raise ValueError("Couldn't find expected result")
    else:
        LOGGER.info("Results matches expected results")


def run_infer(batch_size, init_timeout_s, http_port):
    import numpy as np

    from pytriton.client import ModelClient
    from tests.functional.common.models import ADD_SUB_PYTHON_MODEL

    model_spec = ADD_SUB_PYTHON_MODEL
    a_batch = np.ones((batch_size, 1), dtype=np.float32)
    b_batch = np.ones((batch_size, 1), dtype=np.float32)

    protocol = "http"
    url = f"{protocol}://localhost:{http_port}"
    with ModelClient(url, model_spec.name, init_timeout_s=init_timeout_s) as client:
        result_batch = client.infer_batch(a_batch, b_batch)
        np.testing.assert_allclose(result_batch["add"], a_batch + b_batch)
        np.testing.assert_allclose(result_batch["sub"], a_batch - b_batch)


def main():
    import psutil

    parser = argparse.ArgumentParser(description="short_description")
    parser.add_argument("--timeout-s", required=False, default=300, type=float, help="Timeout for test")
    parser.add_argument("--init-timeout-s", required=False, default=300, type=float, help="Timeout for test")
    parser.add_argument("--seed", type=int, help="PRNG seed", required=False)
    parser.add_argument("--verbose", "-v", action="store_true", help="Provide verbose logs")
    parser.add_argument("--batch-size", type=int, default=32, help="Size of single inference batch")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=DEFAULT_LOG_FORMAT)
    logging.captureWarnings(True)

    start_time_s = time.time()
    wait_time_s = min(args.timeout_s, 5)

    server_script_module = f"{__package__}.server"

    http_port = find_free_port()
    grpc_port = find_free_port()
    server_cmd = [
        "python",
        "-m",
        server_script_module,
        "--http-port",
        str(http_port),
        "--grpc-port",
        str(grpc_port),
        "--seed",
        str(args.seed),
    ]
    if args.verbose:
        server_cmd.append("--verbose")

    with ScriptThread(server_cmd, name="server") as server_thread:
        run_infer(args.batch_size, args.init_timeout_s, http_port)
        elapsed_s = time.time() - start_time_s

        monitoring = ProcessMonitoring(server_thread.process.pid)

        children_processes = server_thread.process.children(recursive=True)
        LOGGER.info(f"Found children processes: {children_processes}")
        LOGGER.info(f"Sending SEGINT to server script process ({server_thread.process})")
        server_thread.process.send_signal(signal.SIGINT)

        def _process_running_and_not_zombie(_process):
            return _process.is_running() and _process.status() != psutil.STATUS_ZOMBIE

        while (
            server_thread.is_alive() or any(_process_running_and_not_zombie(child) for child in children_processes)
        ) and elapsed_s <= args.timeout_s:
            time.sleep(wait_time_s)
            elapsed_s = time.time() - start_time_s
            monitoring.dump_state()

        timeout = elapsed_s >= args.timeout_s and (
            server_thread.is_alive() or any(child.is_running() for child in children_processes)
        )

        if timeout:
            LOGGER.error(f"Timeout occurred (timeout_s={args.timeout_s})")
            sys.exit(-2)
        else:
            LOGGER.info("All processed terminated")

        verify_client_output(server_thread.output)


if __name__ == "__main__":
    main()
