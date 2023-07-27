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
"""Test of fastertransformer_gpt_multinode example"""
import argparse
import logging
import os
import re
import signal
import sys
import time

from tests.utils import DEFAULT_LOG_FORMAT, ScriptThread, search_warning_on_too_verbose_log_level

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "{LOCAL_CONTAINER_REGISTRY}/fastertransformer:{FASTERTRANSFORMER_DOCKER_TAG}",
}


def verify_client_output(client_output):
    expected_pattern = r"Someday I will be able to get a job in the field of medicine"
    output_match = re.search(expected_pattern, client_output, re.MULTILINE)
    output_array = output_match.group(0) if output_match else None
    if not output_array:
        raise ValueError(f"Could not find {expected_pattern} in client output")
    else:
        LOGGER.info(f'Found "{expected_pattern}" in client output')


def main():
    parser = argparse.ArgumentParser(description="short_description")
    parser.add_argument("--timeout-s", required=False, default=300, type=float, help="Timeout for test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format=DEFAULT_LOG_FORMAT)

    os.environ["FT_REPO_DIR"] = "/workspace/FasterTransformer"
    os.environ["PYTHONPATH"] = f"/workspace/FasterTransformer:{os.environ.get('PYTHONPATH', '')}"

    install_cmd = ["bash", "examples/fastertransformer_gpt_multinode/install.sh"]
    with ScriptThread(install_cmd, name="install") as install_thread:
        install_thread.join()

    if install_thread.returncode != 0:
        raise RuntimeError(f"Install thread returned {install_thread.returncode}")

    start_time = time.time()
    elapsed_s = 0
    wait_time_s = min(args.timeout_s, 1)

    server_cmd = ["python", "examples/fastertransformer_gpt_multinode/server.py"]
    client_cmd = ["python", "examples/fastertransformer_gpt_multinode/client.py", "--prompts", "Someday I will"]

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
    assert not search_warning_on_too_verbose_log_level(server_thread.output)


if __name__ == "__main__":
    main()