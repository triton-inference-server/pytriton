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
"""Server with add_sub model"""
import argparse
import logging
import random

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{TEST_CONTAINER_VERSION}-py3",
}


def main():
    from pytriton.triton import Triton, TritonConfig
    from tests.functional.common.models import ADD_SUB_PYTHON_MODEL
    from tests.utils import DEFAULT_LOG_FORMAT, find_free_port

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http-port", type=int, help="HTTP port on which server listens")
    parser.add_argument("--seed", type=int, help="PRNG seed", required=False)
    parser.add_argument("--verbose", "-v", action="store_true", help="Timeout for test")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=DEFAULT_LOG_FORMAT)
    LOGGER.debug(f"CLI args: {args}")

    random.seed(args.seed)

    triton_config = TritonConfig(
        grpc_port=find_free_port(),
        http_port=args.http_port or find_free_port(),
        metrics_port=find_free_port(),
    )
    LOGGER.debug(f"Using {triton_config}")

    with Triton(config=triton_config) as triton:
        model_spec = ADD_SUB_PYTHON_MODEL
        LOGGER.debug(f"Using {model_spec}")
        triton.bind(
            model_name=model_spec.name,
            infer_func=model_spec.create_infer_fn(),
            inputs=model_spec.inputs,
            outputs=model_spec.outputs,
            config=model_spec.model_config,
        )
        triton.serve()


if __name__ == "__main__":
    main()
