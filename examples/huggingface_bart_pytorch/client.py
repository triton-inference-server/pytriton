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
"""Client for BART classifier sample server."""
import argparse
import logging

import numpy as np

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
    parser.add_argument(
        "--init-timeout-s",
        type=float,
        default=600.0,
        help="Server and model ready state timeout in seconds",
        required=False,
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
    logger = logging.getLogger("examples.huggingface_bart_pytorch.client")

    sequence = np.array(["one day I will see the world"])
    labels = np.array(["travel", "cooking", "dancing"])

    sequence = np.char.encode(sequence, "utf-8")
    labels = np.char.encode(labels, "utf-8")

    logger.info(f"Sequence: {sequence}")
    logger.info(f"Labels: {labels}")

    with ModelClient(args.url, "BART", init_timeout_s=args.init_timeout_s) as client:
        result_dict = client.infer_sample(sequence, labels)

    logger.info(f"Result: {result_dict}")


if __name__ == "__main__":
    main()
