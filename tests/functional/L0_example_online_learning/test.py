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
"""Test of online_learning_mnist example"""
import argparse
import logging
import signal
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


def get_accuracy(client_output):
    chunks = client_output.split("Accuracy:")
    if len(chunks) > 1:
        chunks = chunks[-1].split("(")
        chunks = chunks[-1].split("%")
        accuracy = float(chunks[0].strip())
        return accuracy
    else:
        raise ValueError("Couldn't find accuracy in client output")


def check_client_closed_properly(client_thread, timeout):
    if timeout:
        raise ValueError("Client thread timed out")
    if client_thread.is_alive():
        raise ValueError("Client thread is still alive")
    if client_thread.returncode != 0:
        raise ValueError("Client thread exited with non-zero exit code")


def main():
    parser = argparse.ArgumentParser(description="short_description")
    parser.add_argument("--timeout-s", required=False, default=300, type=float, help="Timeout for test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format=DEFAULT_LOG_FORMAT)

    docker_image_with_name = METADATA["image_name"].format(TEST_CONTAINER_VERSION=get_current_container_version())
    verify_docker_image_in_readme_same_as_tested("examples/online_learning_mnist/README.md", docker_image_with_name)

    install_cmd = ["bash", "examples/online_learning_mnist/install.sh"]
    with ScriptThread(install_cmd, name="install") as install_thread:
        install_thread.join()

    if install_thread.returncode != 0:
        raise RuntimeError(f"Install thread returned {install_thread.returncode}")

    start_time = time.time()
    elapsed_s = 0
    wait_time_s = min(args.timeout_s, 1)

    server_cmd = ["python", "examples/online_learning_mnist/server.py"]
    client_train_cmd = ["python", "examples/online_learning_mnist/client_train.py"]
    client_infer_cmd = ["python", "examples/online_learning_mnist/client_infer.py", "--iter", "1"]

    with ScriptThread(server_cmd, name="server") as server_thread:
        try:
            with ScriptThread(client_infer_cmd, name="client") as client_thread:
                while server_thread.is_alive() and client_thread.is_alive() and elapsed_s < args.timeout_s:
                    client_thread.join(timeout=wait_time_s)
                    elapsed_s = time.time() - start_time
                check_client_closed_properly(client_thread, elapsed_s > args.timeout_s)
            acc = get_accuracy(client_thread.output)
            if acc > 40:
                raise ValueError("Accuracy should be close to 0.0")
            LOGGER.info("Accuracy before training ok: %s", acc)

            with ScriptThread(client_train_cmd, name="client") as client_thread:
                while server_thread.is_alive() and client_thread.is_alive() and elapsed_s < args.timeout_s:
                    client_thread.join(timeout=wait_time_s)
                    elapsed_s = time.time() - start_time
                check_client_closed_properly(client_thread, elapsed_s > args.timeout_s)
            LOGGER.info("Training finished.")

            with ScriptThread(client_infer_cmd, name="client") as client_thread:
                while server_thread.is_alive() and client_thread.is_alive() and elapsed_s < args.timeout_s:
                    client_thread.join(timeout=wait_time_s)
                    elapsed_s = time.time() - start_time
                check_client_closed_properly(client_thread, elapsed_s > args.timeout_s)

            acc = get_accuracy(client_thread.output)
            if acc < 90:
                raise ValueError("Accuracy should be close to 100")
            LOGGER.info("Accuracy after training ok: %s", acc)

        finally:
            LOGGER.info("Interrupting server script process")
            if server_thread.process:
                server_thread.process.send_signal(signal.SIGINT)

    if server_thread.returncode not in [0, -2]:  # -2 is returned when process finished after receiving SIGINT signal
        raise ValueError(f"Server returned {server_thread.returncode}")

    timeout = elapsed_s >= args.timeout_s and client_thread.is_alive() and server_thread.is_alive()
    if timeout:
        LOGGER.error(f"Timeout occurred (timeout_s={args.timeout_s})")
        sys.exit(-2)


if __name__ == "__main__":
    main()
