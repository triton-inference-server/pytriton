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
    import pytest

    from pytriton.check.utils import DEFAULT_LOG_FORMAT, find_free_port
    from pytriton.client import ModelClient
    from pytriton.client.exceptions import PyTritonClientInferenceServerError
    from pytriton.decorators import batch
    from pytriton.model_config import ModelConfig, Tensor
    from pytriton.triton import Triton, TritonConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-timeout-s", required=False, default=300, type=float, help="Timeout for server and models initialization"
    )
    parser.add_argument("--seed", type=int, help="PRNG seed", required=False)
    parser.add_argument("--verbose", "-v", action="store_true", help="Timeout for test")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=DEFAULT_LOG_FORMAT)
    logging.captureWarnings(True)
    LOGGER.debug("CLI args: %s", args)

    random.seed(args.seed)

    triton_config = TritonConfig(grpc_port=find_free_port(), http_port=find_free_port(), metrics_port=find_free_port())
    LOGGER.debug("Using %s", triton_config)

    @batch
    def _throw_division_error(**_):
        return 2 / 0

    with Triton(config=triton_config) as triton:
        triton.bind(
            model_name="proxy",
            infer_func=_throw_division_error,
            inputs=[Tensor(dtype=np.float32, shape=(-1,))],
            outputs=[Tensor(dtype=np.float32, shape=(-1,))],
            config=ModelConfig(max_batch_size=128),
        )
        triton.run()

        batch_size = 6
        input1 = np.arange(batch_size * 10, batch_size * 10 + batch_size, 1).reshape(batch_size, 1).astype(np.float32)

        protocol = random.choice(["http", "grpc"])
        url = f"{protocol}://localhost:{getattr(triton_config, f'{protocol}_port')}"
        with ModelClient(url, "proxy", init_timeout_s=args.init_timeout_s) as client:
            with pytest.raises(PyTritonClientInferenceServerError, match="division by zero"):
                client.infer_batch(input1)


if __name__ == "__main__":
    main()
