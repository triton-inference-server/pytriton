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
"""e2e tests inference on add_sub model"""
import argparse
import logging
import random
import re
import sys
import time

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{version}-py3",
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
            LOGGER.info(f'Found "{expected_pattern}" in client output')


def main():
    from pytriton.client import ModelClient
    from pytriton.triton import Triton, TritonConfig
    from tests.functional.common.models import BART_CLASSIFIER_PYTORCH_MODEL
    from tests.utils import DEFAULT_LOG_FORMAT, ScriptThread, find_free_port

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-timeout-s", required=False, default=300, type=float, help="Timeout for server and models initialization"
    )
    parser.add_argument("--timeout-s", required=False, default=300, type=float, help="Timeout for perf_analyzer run")
    parser.add_argument("--input-data-path", help="Path to perf_analyzer input data")
    parser.add_argument("--seed", type=int, help="PRNG seed", required=False)
    parser.add_argument("--verbose", "-v", action="store_true", help="Timeout for test")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=DEFAULT_LOG_FORMAT)
    LOGGER.debug(f"CLI args: {args}")

    random.seed(args.seed)

    triton_config = TritonConfig(grpc_port=find_free_port(), http_port=find_free_port(), metrics_port=find_free_port())
    LOGGER.debug(f"Using {triton_config}")

    with Triton(config=triton_config) as triton:
        model_spec = BART_CLASSIFIER_PYTORCH_MODEL
        LOGGER.debug(f"Using {model_spec}")
        triton.bind(
            model_name=model_spec.name,
            infer_func=model_spec.create_infer_fn(),
            inputs=model_spec.inputs,
            outputs=model_spec.outputs,
            config=model_spec.model_config,
        )
        triton.run()

        protocol = random.choice(["http", "grpc"])
        protocol_port = getattr(triton_config, f"{protocol}_port")
        url = f"{protocol}://localhost:{protocol_port}"

        ModelClient(url, model_spec.name).wait_for_model(timeout_s=args.init_timeout_s)
        client_cmd = [
            "perf_analyzer",
            "-u",
            f"localhost:{protocol_port}",
            "-i",
            protocol,
            "-m",
            model_spec.name,
            "--measurement-mode",
            "count_windows",
            "--measurement-request-count",
            "100",
            "--input-data",
            args.input_data_path,
            "--concurrency-range",
            "4:16:4",
            "-v",
        ]
        with ScriptThread(client_cmd, name="client") as client_thread:
            start_time = time.time()
            elapsed_s = 0
            while client_thread.is_alive() and elapsed_s < args.timeout_s:
                client_thread.join(timeout=1)
                elapsed_s = time.time() - start_time

    if client_thread.returncode != 0:
        raise RuntimeError(f"Client returned {client_thread.returncode}")

    timeout = elapsed_s >= args.timeout_s and client_thread.is_alive()
    if timeout:
        LOGGER.error(f"Timeout occurred (timeout_s={args.timeout_s})")
        sys.exit(-2)

    verify_client_output(client_thread.output)


if __name__ == "__main__":
    main()
