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
"""Tests errors passing e2e"""
import argparse
import logging
import random

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{TEST_CONTAINER_VERSION}-py3",
    "platforms": ["amd64", "arm64"],
}


def main():
    import numpy as np

    from pytriton.client import ModelClient
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
    LOGGER.debug(f"CLI args: {args}")

    class _InferFuncWrapper:
        def __init__(self):
            self.call_count = 0

        @batch
        def infer_func(self, **inputs):
            a_batch, b_batch = inputs.values()
            add_batch = a_batch + b_batch
            sub_batch = a_batch - b_batch
            self.call_count += 1
            return {"add": add_batch, "sub": sub_batch}

    random.seed(args.seed)
    infer_func_wrapper = _InferFuncWrapper()

    triton_config = TritonConfig(
        grpc_port=find_free_port(),
        http_port=find_free_port(),
        metrics_port=find_free_port(),
        cache_config=[f"local,size={1024 * 1024}"],  # 1 MB
    )
    LOGGER.debug(f"Using {triton_config}")
    with Triton(config=triton_config) as triton:
        triton.bind(
            model_name="AddSub",
            infer_func=infer_func_wrapper.infer_func,
            inputs=(
                Tensor(dtype=np.float32, shape=(-1,)),
                Tensor(dtype=np.float32, shape=(-1,)),
            ),
            outputs=(
                Tensor(name="add", dtype=np.float32, shape=(-1,)),
                Tensor(name="sub", dtype=np.float32, shape=(-1,)),
            ),
            config=ModelConfig(max_batch_size=16, response_cache=True),
        )
        triton.run()

        batch_size = 16
        a_batch = np.ones((batch_size, 1), dtype=np.float32)
        b_batch = np.ones((batch_size, 1), dtype=np.float32)

        protocol = random.choice(["http", "grpc"])
        protocol_port = getattr(triton_config, f"{protocol}_port")
        url = f"{protocol}://localhost:{protocol_port}"
        with ModelClient(url, "AddSub", init_timeout_s=args.init_timeout_s) as client:
            for idx in range(10):
                LOGGER.info(f"Sending request {idx + 1}")
                result_batch = client.infer_batch(a_batch, b_batch)
                LOGGER.info(f"Response obtained for {idx + 1}. Number of outputs: {len(result_batch)}")

            LOGGER.info("Validating response.")
            np.testing.assert_allclose(result_batch["add"], a_batch + b_batch)
            np.testing.assert_allclose(result_batch["sub"], a_batch - b_batch)

    LOGGER.info(f"Infer function requests count: {infer_func_wrapper.call_count}")

    assert infer_func_wrapper.call_count == 1


if __name__ == "__main__":
    main()
