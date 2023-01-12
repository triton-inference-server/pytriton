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
"""Client for linear_random example."""
import argparse
import logging

import torch  # pytype: disable=import-error

from pytriton.client import ModelClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="localhost",
        help=(
            "Url to Triton server (ex. grpc://localhost:8001)."
            "HTTP protocol with default port is used if parameter is not provided"
        ),
        required=False,
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
    logger = logging.getLogger("examples.linear_random_pytorch.client")

    input1_batch = torch.randn(128, 20).cpu().detach().numpy()

    logger.info(f"Input: {input1_batch.tolist()}")

    with ModelClient(args.url, "Linear") as client:
        logger.info("Sending request")
        result_dict = client.infer_batch(input1_batch)

    for output_name, output_batch in result_dict.items():
        logger.info(f"{output_name}: {output_batch.tolist()}")


if __name__ == "__main__":
    main()
