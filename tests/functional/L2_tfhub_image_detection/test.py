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
"""
Runs longer (10h) inference session over image detector model
"""
import argparse
import logging
import random

METADATA = {
    "image_name": "nvcr.io/nvidia/tensorflow:{TEST_CONTAINER_VERSION}-tf2-py3",
    "shared_memory_size_mb": 640,
}


def main():
    from tests.functional.common.tests.tfhub_image_detection import tfhub_image_detection
    from tests.utils import DEFAULT_LOG_FORMAT

    parser = argparse.ArgumentParser(description="short_description")
    parser.add_argument(
        "--test-time-s",
        required=False,
        default=36000,
        type=int,
        help="Time for how long the test should be run.",
    )
    parser.add_argument(
        "--init-timeout-s",
        required=False,
        default=300,
        type=int,
        help="Timeout for server and models initialization",
    )
    parser.add_argument(
        "--batch-size",
        required=False,
        default=128,
        type=int,
        help="Maximal batch size used for model deployment",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="PRNG seed",
        required=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Provide verbose logs",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=DEFAULT_LOG_FORMAT)

    tfhub_image_detection(
        test_time_s=args.test_time_s,
        init_timeout_s=args.init_timeout_s,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
