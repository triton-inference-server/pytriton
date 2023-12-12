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
"""Test of vllm example"""
import argparse
import logging
import signal
import sys
import time

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{TEST_CONTAINER_VERSION}-py3",
}


def _wait_for_model_ready(url, model_name, timeout_s=300):
    from pytriton.client.utils import create_client_from_url

    start_time = time.time()
    elapsed_s = 0
    with create_client_from_url(url) as client:
        is_model_ready = False
        try:
            is_model_ready = client.is_model_ready(model_name)
        except Exception:
            pass

        while not is_model_ready and elapsed_s < timeout_s:
            time.sleep(1.0)
            try:
                is_model_ready = client.is_model_ready("gpt2")
            except Exception:
                pass
            elapsed_s = time.time() - start_time


def verify_client_output(client_output):
    expected_matches = ["San Francisco is a city of more than 1 million people"]
    for expected_match in expected_matches:
        if expected_match not in client_output:
            raise ValueError(f"Couldn't find expected result: {expected_match}. Got: {client_output}")
    LOGGER.info("Results matches expected results")


def main():
    from tests.utils import (
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
    verify_docker_image_in_readme_same_as_tested("examples/vllm/README.md", docker_image_with_name)

    install_cmd = ["bash", "examples/vllm/install.sh"]
    with ScriptThread(install_cmd, name="install") as install_thread:
        install_thread.join()

    if install_thread.returncode != 0:
        raise RuntimeError(f"Install thread returned {install_thread.returncode}")

    start_time = time.time()
    elapsed_s = 0
    wait_time_s = min(args.timeout_s, 1)

    server_cmd = ["python", "examples/vllm/server.py", "--model", "gpt2"]
    client_cmd = ["examples/vllm/client.sh", "gpt2"]
    client_streaming_cmd = ["examples/vllm/client_streaming.sh", "gpt2"]

    with ScriptThread(server_cmd, name="server") as server_thread:
        _wait_for_model_ready("localhost:8000", "gpt2", timeout_s=args.timeout_s)
        with ScriptThread(client_cmd, name="client") as client_thread:
            with ScriptThread(client_streaming_cmd, name="client-streaming") as client_streaming_thread:
                while (
                    all(
                        [
                            server_thread.is_alive(),
                            client_thread.is_alive(),
                            client_streaming_thread.is_alive(),
                        ]
                    )
                    and elapsed_s < args.timeout_s
                ):
                    client_thread.join(timeout=wait_time_s)
                    client_streaming_thread.join(timeout=wait_time_s)
                    elapsed_s = time.time() - start_time

        LOGGER.info("Interrupting server script process")
        if server_thread.process:
            server_thread.process.send_signal(signal.SIGINT)
            server_thread.join(timeout=wait_time_s)

    if client_thread.returncode != 0:
        raise RuntimeError(f"Client returned {client_thread.returncode}")
    if client_streaming_thread.returncode != 0:
        raise RuntimeError(f"Client streaming returned {client_streaming_thread.returncode}")
    if server_thread.returncode not in [0, -2]:  # -2 is returned when process finished after receiving SIGINT signal
        raise RuntimeError(f"Server returned {server_thread.returncode}")

    timeout = elapsed_s >= args.timeout_s and client_thread.is_alive() and server_thread.is_alive()
    if timeout:
        LOGGER.error(f"Timeout occurred (timeout_s={args.timeout_s})")
        sys.exit(-2)

    verify_client_output(client_thread.output)
    verify_client_output(client_streaming_thread.output)
    assert not search_warning_on_too_verbose_log_level(server_thread.output)


if __name__ == "__main__":
    main()
