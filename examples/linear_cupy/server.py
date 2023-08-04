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
"""Server with simple python model performing adding and subtract operation."""
import logging

import cupy as cp  # pytype: disable=import-error
import numpy as np

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

logger = logging.getLogger("examples.linear_cupy.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

VECTOR_SIZE = 10


class LinearModel:
    def __init__(self):
        self.alpha = 2
        self.beta = cp.arange(VECTOR_SIZE)

    @batch
    def linear(self, **inputs):
        u_batch, v_batch = inputs.values()
        u_batch_cp, v_batch_cp = cp.asarray(u_batch), cp.asarray(v_batch)
        lin = u_batch_cp * self.alpha + v_batch_cp + self.beta
        return {"result": cp.asnumpy(lin)}


with Triton() as triton:
    logger.info("Loading linear model")
    lin_model = LinearModel()
    triton.bind(
        model_name="Linear",
        infer_func=lin_model.linear,
        inputs=[
            Tensor(dtype=np.float64, shape=(VECTOR_SIZE,)),
            Tensor(dtype=np.float64, shape=(VECTOR_SIZE,)),
        ],
        outputs=[
            Tensor(name="result", dtype=np.float64, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128),
        strict=True,
    )
    logger.info("Serving model")
    triton.serve()
