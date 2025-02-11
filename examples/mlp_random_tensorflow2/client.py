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
"""Client for mlp_random example."""

import logging

import numpy as np

from pytriton.client import ModelClient

logger = logging.getLogger("examples.mlp_random_tensorflow2.client")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

batch_size = 16
image_size = (224, 224, 3)
images_batch = np.random.uniform(size=(batch_size,) + image_size).astype(np.float32)

logger.info("Input: %s", images_batch)

with ModelClient("localhost", "MLP") as client:
    logger.info("Sending request")
    result_dict = client.infer_batch(images_batch)

logger.info("results: %s", result_dict)
