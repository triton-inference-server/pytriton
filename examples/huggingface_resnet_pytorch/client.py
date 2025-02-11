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
"""Client for ResNet50 classifier sample server."""

import argparse
import io
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from datasets import load_dataset  # pytype: disable=import-error

from pytriton.client import ModelClient

logger = logging.getLogger("examples.huggingface_bart_pytorch.client")


def infer_model(thread_idx, args, dataset):
    with ModelClient(args.url, "ResNet", init_timeout_s=args.init_timeout_s) as client:
        image = dataset["image"][0]
        logger.info("Image: %s", image)

        output = io.BytesIO()
        image.save(output, format="JPEG")
        image = np.frombuffer(output.getbuffer(), dtype=np.uint8)

        logger.info("Running inference requests in thread %d.", thread_idx)

        for req_idx in range(1, args.iterations + 1):
            logger.debug("Sending request (%d) in thread %d.", req_idx, thread_idx)
            result_data = client.infer_sample(image)
            logger.debug("Result: %s for request (%d) in thread %d.", result_data, req_idx, thread_idx)

        logger.info("Last result: %s for request (%d) in thread %d.", result_data, req_idx, thread_idx)


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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Number of concurrent requests.",
        required=False,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of requests per client.",
        required=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    dataset = load_dataset("huggingface/cats-image", split="test")
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        running_tasks = [
            executor.submit(infer_task, idx, args, dataset)
            for idx, infer_task in enumerate([infer_model] * args.concurrency)
        ]
        for running_task in running_tasks:
            running_task.result()


if __name__ == "__main__":
    main()
