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
"""Tests scenario for each of signals [SIGINT, SIGTERM]:
- starting server
- wait for server to be ready
- checks if resources are obtained
  - shared memory
- sends signal to server script
- wait server shutdown (with timeout)
- checks if resources are released
"""
import argparse
import logging
import pathlib
import signal
import sys
import time

from tests.utils import ProcessMonitoring  # pytype: disable=import-error
from tests.utils import DEFAULT_LOG_FORMAT, ScriptThread, find_free_port

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])

METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{TEST_CONTAINER_VERSION}-py3",
}


def _run_infer(url, init_timeout_s, batch_size):
    import numpy as np

    from pytriton.client import ModelClient
    from tests.functional.common.models import ADD_SUB_PYTHON_MODEL

    model_spec = ADD_SUB_PYTHON_MODEL
    a_batch = np.ones((batch_size, 1), dtype=np.float32)
    b_batch = np.ones((batch_size, 1), dtype=np.float32)

    with ModelClient(url, model_spec.name, init_timeout_s=init_timeout_s) as client:
        result_batch = client.infer_batch(a_batch, b_batch)
        np.testing.assert_allclose(result_batch["add"], a_batch + b_batch)
        np.testing.assert_allclose(result_batch["sub"], a_batch - b_batch)


def _check_resources_allocated(initial_shared_memory_files):
    shared_memory_files = sorted(set(pathlib.Path("/dev/shm").rglob("*")) - set(initial_shared_memory_files))
    assert shared_memory_files, shared_memory_files


def _check_resources_released(initial_shared_memory_files):
    shared_memory_files = sorted(set(pathlib.Path("/dev/shm").rglob("*")) - set(initial_shared_memory_files))
    assert not shared_memory_files, shared_memory_files


def _run_test(init_timeout_s, verbose, seed, signal_value, test_timeout_s):
    import psutil

    start_time_s = time.time()
    wait_time_s = min(test_timeout_s, 5)

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
        str(seed),
    ]
    if verbose:
        server_cmd.append("--verbose")

    initial_shared_memory_files = list(pathlib.Path("/dev/shm").rglob("*"))
    with ScriptThread(server_cmd, name="server") as server_thread:
        url = f"http://localhost:{http_port}"
        _run_infer(url, init_timeout_s, batch_size=32)
        monitoring = ProcessMonitoring(server_thread.process.pid)
        elapsed_s = time.time() - start_time_s

        children_processes = server_thread.process.children(recursive=True)
        LOGGER.info(f"Found children processes: {children_processes}")

        _check_resources_allocated(initial_shared_memory_files)
        LOGGER.info(f"Sending {signal_value} to server script process ({server_thread.process})")
        server_thread.process.send_signal(signal_value)

        LOGGER.info("Waiting for server script and all its children processes to finish")

        def _process_running_and_not_zombie(_process):
            return _process.is_running() and _process.status() != psutil.STATUS_ZOMBIE

        while (
            server_thread.is_alive() or any(_process_running_and_not_zombie(child) for child in children_processes)
        ) and elapsed_s <= test_timeout_s:
            time.sleep(wait_time_s)
            elapsed_s = time.time() - start_time_s
            monitoring.dump_state()

        timeout = elapsed_s >= test_timeout_s and (
            server_thread.is_alive() or any(child.is_running() for child in children_processes)
        )

        if timeout:
            LOGGER.error(f"Timeout occurred (timeout_s={test_timeout_s})")
            sys.exit(-2)
        else:
            LOGGER.info("All processed terminated")

    _check_resources_released(initial_shared_memory_files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-timeout-s", required=False, default=300, type=float, help="Timeout for each subtest performance"
    )
    parser.add_argument("--init-timeout-s", required=False, default=300, type=float, help="Timeout for test")
    parser.add_argument("--seed", type=int, help="PRNG seed", required=False)
    parser.add_argument("--verbose", "-v", action="store_true", help="Provide verbose logs")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=DEFAULT_LOG_FORMAT)

    _run_test(args.init_timeout_s, args.verbose, args.seed, signal.SIGINT, args.test_timeout_s)
    _run_test(args.init_timeout_s, args.verbose, args.seed, signal.SIGTERM, args.test_timeout_s)


if __name__ == "__main__":
    main()
