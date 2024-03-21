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
"""Client for identity_python sample server."""

import logging
import random

import numpy as np

from pytriton.client import ModelClient

logger = logging.getLogger("examples.identity_python.client")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

batch_size = 4
input1_batch = [[random.random(), random.random(), random.random(), random.random()]] * batch_size
input2_batch = [[b"\xff\x00\x00\x00"]] * batch_size

logger.info(f"INPUT_1: {input1_batch}")
logger.info(f"INPUT_2: {input2_batch}")

input1_batch = np.array(input1_batch, dtype=np.float64)
input2_batch = np.array(input2_batch, dtype=object)  # use dtype=object to avoid trimming of `\x00` bytes by numpy

with ModelClient("localhost", "Identity") as client:
    logger.info("Sending request")
    result_dict = client.infer_batch(input1_batch, input2_batch)
    logger.info(f"results: {result_dict}")

for output_name, output_batch in result_dict.items():
    logger.info(f"{output_name}: {output_batch.tolist()}")
