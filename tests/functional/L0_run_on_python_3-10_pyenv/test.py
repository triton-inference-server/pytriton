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
"""e2e tests inference on add_sub model"""

import argparse
import logging
import random

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "ubuntu:22.04",
    "platforms": ["amd64", "arm64"],
}


def main():
    import numpy as np

    from pytriton.check.utils import DEFAULT_LOG_FORMAT, find_free_port
    from pytriton.client import ModelClient
    from pytriton.triton import Triton, TritonConfig
    from tests.functional.common.models import ADD_SUB_PYTHON_MODEL

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-timeout-s", required=False, default=300, type=float, help="Timeout for server and models initialization"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Size of single inference batch")
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

    with Triton(config=triton_config) as triton:
        model_spec = ADD_SUB_PYTHON_MODEL
        LOGGER.debug("Using %s", model_spec)
        triton.bind(
            model_name=model_spec.name,
            infer_func=model_spec.create_infer_fn(),
            inputs=model_spec.inputs,
            outputs=model_spec.outputs,
            config=model_spec.model_config,
        )
        triton.run()

        batch_size = args.batch_size
        a_batch = np.ones((batch_size, 1), dtype=np.float32)
        b_batch = np.ones((batch_size, 1), dtype=np.float32)

        protocol = random.choice(["http", "grpc"])
        protocol_port = getattr(triton_config, f"{protocol}_port")
        url = f"{protocol}://localhost:{protocol_port}"
        with ModelClient(url, model_spec.name, init_timeout_s=args.init_timeout_s) as client:
            result_batch = client.infer_batch(a_batch, b_batch)
            np.testing.assert_allclose(result_batch["add"], a_batch + b_batch)
            np.testing.assert_allclose(result_batch["sub"], a_batch - b_batch)


if __name__ == "__main__":
    main()
