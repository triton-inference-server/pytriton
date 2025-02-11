#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
"""Client for simple_python_remote_mode sample server."""

import logging

import numpy as np

from pytriton.client import ModelClient

logger = logging.getLogger("examples.simple_python_remote_mode.client")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

batch_size = 2
a_batch = np.array([[1.0], [2.0]], dtype=np.float32)
b_batch = np.array([[2.0], [3.0]], dtype=np.float32)

logger.info("a: %s", a_batch.tolist())
logger.info("b: %s", b_batch.tolist())

with ModelClient("localhost", "AddSub", model_version="1") as client:
    logger.info("Sending inference request")
    result_batch = client.infer_batch(a_batch, b_batch)

    for output_name, data_batch in result_batch.items():
        logger.info("%s: %s", output_name, data_batch.tolist())

with ModelClient("localhost", "Mul") as client:
    logger.info("Sending inference request")
    result_batch = client.infer_batch(a_batch, b_batch)

    for output_name, data_batch in result_batch.items():
        logger.info("%s: %s", output_name, data_batch.tolist())

with ModelClient("localhost", "Power") as client:
    logger.info("Sending inference request")
    result_batch = client.infer_batch(a_batch, b_batch)

    for output_name, data_batch in result_batch.items():
        logger.info("%s: %s", output_name, data_batch.tolist())
