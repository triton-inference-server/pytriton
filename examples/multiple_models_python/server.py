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
"""Example with multiple models served on single Triton server."""
import logging

import numpy as np

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

logger = logging.getLogger("examples.multiple_models_python.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


@batch
def _multiply2(multiplicand):
    product = multiplicand * 2.0
    return [product]


@batch
def _multiply4(multiplicand):
    product = multiplicand * 4.0
    return [product]


with Triton() as triton:
    logger.info("Loading Multiply2 model")
    triton.bind(
        model_name="Multiply2",
        infer_func=_multiply2,
        inputs=[
            Tensor(name="multiplicand", dtype=np.float32, shape=(-1,)),
        ],
        outputs=[
            Tensor(name="product", dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=8),
    )
    logger.info("Loading Multiply4 model")
    triton.bind(
        model_name="Multiply4",
        infer_func=_multiply4,
        inputs=[
            Tensor(name="multiplicand", dtype=np.float32, shape=(-1,)),
        ],
        outputs=[
            Tensor(name="product", dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=8),
    )
    triton.serve()
