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
"""Client for add_sub_python sample server."""

import logging

import numpy as np

from pytriton.client import ModelClient

logger = logging.getLogger("examples.linear_cupy.client")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

VECTOR_SIZE = 10
BATCH_SIZE = 2

u_batch = np.ones((BATCH_SIZE, VECTOR_SIZE), dtype=np.float64)
v_batch = np.ones((BATCH_SIZE, VECTOR_SIZE), dtype=np.float64)

logger.info("u: %s", u_batch.tolist())
logger.info("v: %s", v_batch.tolist())

with ModelClient("localhost", "Linear") as client:
    logger.info("Sending inference request")
    result_batch = client.infer_batch(u_batch, v_batch)

for output_name, data_batch in result_batch.items():
    logger.info("%s: %s", output_name, data_batch.tolist())
