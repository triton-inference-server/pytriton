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
"""Tests errors passing e2e"""
import argparse
import io
import logging
import random

from tests.utils import search_warning_on_too_verbose_log_level

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{TEST_CONTAINER_VERSION}-py3",
}


def main():
    import threading

    import numpy as np

    from pytriton.client import ModelClient
    from pytriton.client.utils import wait_for_server_ready
    from pytriton.decorators import batch
    from pytriton.model_config import ModelConfig, Tensor
    from pytriton.triton import Triton, TritonConfig
    from tests.utils import DEFAULT_LOG_FORMAT, find_free_port

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-timeout-s", required=False, default=300, type=float, help="Timeout for server and models initialization"
    )
    parser.add_argument(
        "--shutdown-timeout-s",
        required=False,
        default=300,
        type=float,
        help="Timeout for server to shutdown on PyTritonUnrecoverableError",
    )
    parser.add_argument("--seed", type=int, help="PRNG seed", required=False)
    parser.add_argument("--verbose", "-v", action="store_true", help="Timeout for test")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(level=log_level, format=DEFAULT_LOG_FORMAT)

    handler = logging.StreamHandler(io.StringIO())
    handler.setLevel(log_level)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    LOGGER.debug(f"CLI args: {args}")

    random.seed(args.seed)

    triton_config = TritonConfig(
        grpc_port=find_free_port(), http_port=find_free_port(), metrics_port=find_free_port(), log_verbose=4
    )
    LOGGER.debug(f"Using {triton_config}")

    @batch
    def infer_fn(**inputs):
        return inputs

    with Triton(config=triton_config) as triton:
        triton.bind(
            model_name="proxy",
            infer_func=infer_fn,
            inputs=[Tensor(dtype=np.float32, shape=(-1,))],
            outputs=[Tensor(dtype=np.float32, shape=(-1,))],
            config=ModelConfig(max_batch_size=128),
        )
        triton.run()
        client = ModelClient(f"http://localhost:{triton_config.http_port}", "Dummy")
        condition = threading.Condition(threading.RLock())
        with condition:
            wait_for_server_ready(client._general_client, timeout_s=args.init_timeout_s, condition=condition)

    # obtain logs from handler
    logs = handler.stream.getvalue()
    assert search_warning_on_too_verbose_log_level(logs)


if __name__ == "__main__":
    main()
