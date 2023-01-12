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
"""Example client script for multiple_models example."""
import logging

import numpy as np

from pytriton.client import ModelClient

logger = logging.getLogger("examples.multiple_models_python.client")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

batch_size = 2
a_batch = np.ones((batch_size, 1), dtype=np.float32)

logger.info(f"a: {a_batch.tolist()}")

with ModelClient("localhost", "Multiply2") as client2:
    with ModelClient("localhost", "Multiply4") as client4:
        result2_batch = client2.infer_batch(a_batch)
        result4_batch = client4.infer_batch(a_batch)

for output_name, data_batch in result2_batch.items():
    logger.info(f"Multiply2/{output_name}: {data_batch.tolist()}")
for output_name, data_batch in result4_batch.items():
    logger.info(f"Multiply4/{output_name}: {data_batch.tolist()}")
