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
"""e2e tests inference on triton with multiple models"""
import argparse
import logging
import random

import numpy as np

METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{version}-py3",
}


def main():
    from pytriton.client import ModelClient
    from pytriton.triton import Triton, TritonConfig
    from tests.functional.common.models import ADD_SUB_PYTHON_MODEL, IDENTITY_PYTHON_MODEL
    from tests.utils import DEFAULT_LOG_FORMAT, find_free_port

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-timeout-s", required=False, default=300, type=float, help="Timeout for server and models initialization"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Size of single inference batch")
    parser.add_argument("--seed", type=int, help="PRNG seed", required=False)
    parser.add_argument("--verbose", "-v", action="store_true", help="Timeout for test")
    logger = logging.getLogger((__package__ or "main").split(".")[-1])
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=DEFAULT_LOG_FORMAT)
    logger.debug(f"CLI args: {args}")

    random.seed(args.seed)

    triton_config = TritonConfig(grpc_port=find_free_port(), http_port=find_free_port(), metrics_port=find_free_port())
    logger.debug(f"Using {triton_config}")

    with Triton(config=triton_config) as triton:
        model1_spec = ADD_SUB_PYTHON_MODEL
        logger.debug(f"Using {model1_spec}")
        triton.bind(
            model_name=model1_spec.name,
            infer_func=model1_spec.create_infer_fn(),
            inputs=model1_spec.inputs,
            outputs=model1_spec.outputs,
            config=model1_spec.model_config,
        )
        model2_spec = IDENTITY_PYTHON_MODEL
        logger.debug(f"Using {model2_spec}")
        triton.bind(
            model_name=model2_spec.name,
            infer_func=model2_spec.create_infer_fn(),
            inputs=model2_spec.inputs,
            outputs=model2_spec.outputs,
            config=model2_spec.model_config,
        )
        triton.run()

        batch_size = args.batch_size
        a_batch = np.ones((batch_size, 1), dtype=np.float32)
        b_batch = np.ones((batch_size, 1), dtype=np.float32)

        protocol = random.choice(["http", "grpc"])
        protocol_port = getattr(triton_config, f"{protocol}_port")
        url = f"{protocol}://localhost:{protocol_port}"

        with ModelClient(url, model1_spec.name, init_timeout_s=args.init_timeout_s) as client1, ModelClient(
            url, model2_spec.name, init_timeout_s=args.init_timeout_s
        ) as client2:
            result1_batch = client1.infer_batch(a_batch, b_batch)
            result2_batch = client2.infer_batch(a_batch)

            np.testing.assert_allclose(result1_batch["add"], a_batch + b_batch)
            np.testing.assert_allclose(result1_batch["sub"], a_batch - b_batch)
            np.testing.assert_allclose(result2_batch["identity"], a_batch)


if __name__ == "__main__":
    main()
