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
"""Test of add_sub_vecrtex_ai example"""

import argparse
import logging
import signal
import sys
import time

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{TEST_CONTAINER_VERSION}-py3",
    "platforms": ["amd64", "arm64"],
}


def verify_client_output(client_output):
    output1_match = "add: [[2.0], [2.0]]" in client_output
    output2_match = "sub: [[0.0], [0.0]]" in client_output
    if not (output1_match and output2_match):
        raise ValueError("Couldn't find expected result")
    else:
        LOGGER.info("Results matches expected results")


def verify_vertex_ai_endpoint_output(client_output):
    output1_match = '{"name":"add","datatype":"FP32","shape":[1,4],"data":[2.0,4.0,6.0,8.0]}' in client_output
    output2_match = '{"name":"sub","datatype":"FP32","shape":[1,4],"data":[0.0,0.0,0.0,0.0]}' in client_output
    if not (output1_match and output2_match):
        raise ValueError("Couldn't find expected result in Vertex AI endpoint output")
    else:
        LOGGER.info("Results matches expected results in Vertex AI endpoint output")


def verify_vertex_ai_health_endpoint_output(client_output):
    output_match = "HTTP/1.1 200 OK" in client_output
    if not output_match:
        raise ValueError("Couldn't find expected result in Vertex AI health endpoint output")
    else:
        LOGGER.info("Results matches expected results in Vertex AI health endpoint output")


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
    verify_docker_image_in_readme_same_as_tested("examples/add_sub_vertex_ai/README.md", docker_image_with_name)

    install_cmd = ["bash", "examples/add_sub_vertex_ai/install.sh"]
    with ScriptThread(install_cmd, name="install") as install_thread:
        install_thread.join()

    if install_thread.returncode != 0:
        raise RuntimeError(f"Install thread returned {install_thread.returncode}")

    start_time = time.time()
    elapsed_s = 0
    wait_time_s = min(args.timeout_s, 1)

    server_cmd = ["python", "examples/add_sub_vertex_ai/server.py"]
    client_cmd = ["python", "examples/add_sub_vertex_ai/client.py"]
    vertex_ai_infer_cmd = ["./examples/add_sub_vertex_ai/infer_vertex_ai_endpoint.sh"]
    vertex_ai_health_cmd = ["./examples/add_sub_vertex_ai/health_vertex_ai_endpoint.sh"]

    with ScriptThread(server_cmd, name="server") as server_thread:
        with ScriptThread(client_cmd, name="client") as client_thread:
            while server_thread.is_alive() and client_thread.is_alive() and elapsed_s < args.timeout_s:
                client_thread.join(timeout=wait_time_s)
                elapsed_s = time.time() - start_time
        with ScriptThread(vertex_ai_infer_cmd, name="vertex_ai_infer") as vertex_ai_infer_thread:
            while vertex_ai_infer_thread.is_alive() and elapsed_s < args.timeout_s:
                vertex_ai_infer_thread.join(timeout=wait_time_s)
                elapsed_s = time.time() - start_time
        with ScriptThread(vertex_ai_health_cmd, name="vertex_ai_health") as vertex_ai_health_thread:
            while vertex_ai_health_thread.is_alive() and elapsed_s < args.timeout_s:
                vertex_ai_health_thread.join(timeout=wait_time_s)
                elapsed_s = time.time() - start_time

        LOGGER.info("Interrupting server script process")
        if server_thread.process:
            server_thread.process.send_signal(signal.SIGINT)

    if client_thread.returncode != 0:
        raise RuntimeError(f"Client returned {client_thread.returncode}")
    if vertex_ai_infer_thread.returncode != 0:
        raise RuntimeError(f"Vertex AI infer returned {vertex_ai_infer_thread.returncode}")
    if vertex_ai_health_thread.returncode != 0:
        raise RuntimeError(f"Vertex AI health returned {vertex_ai_health_thread.returncode}")
    if server_thread.returncode not in [0, -2]:  # -2 is returned when process finished after receiving SIGINT signal
        raise RuntimeError(f"Server returned {server_thread.returncode}")

    timeout = elapsed_s >= args.timeout_s and client_thread.is_alive() and server_thread.is_alive()
    if timeout:
        LOGGER.error(f"Timeout occurred (timeout_s={args.timeout_s})")
        sys.exit(-2)

    LOGGER.info("Outputs:")
    LOGGER.info(client_thread.output)
    LOGGER.info(vertex_ai_infer_thread.output)
    LOGGER.info(vertex_ai_health_thread.output)

    verify_client_output(client_thread.output)
    verify_vertex_ai_endpoint_output(vertex_ai_infer_thread.output)
    verify_vertex_ai_health_endpoint_output(vertex_ai_health_thread.output)

    assert not search_warning_on_too_verbose_log_level(server_thread.output)


if __name__ == "__main__":
    main()
