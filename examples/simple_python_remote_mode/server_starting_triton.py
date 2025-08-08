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
"""Server starting triton with simple python model performing adding and subtract operation."""

import logging
import os

import numpy as np

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonSecurityConfig

logger = logging.getLogger("examples.simple_python_remote_mode.server_starting_triton")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


@batch
def _add_sub(**inputs):
    a_batch, b_batch = inputs.values()
    add_batch = a_batch + b_batch
    sub_batch = a_batch - b_batch
    return {"add": add_batch, "sub": sub_batch}


with Triton(security_config=TritonSecurityConfig(access_token=os.getenv("TRITON_ACCESS_TOKEN"))) as triton:
    logger.info("Loading and serve AddSub model")

    # triton.bind() is optional here (you can use Triton class for starting server only without binding any model
    # and then use RemoteTriton class from separate script to bind model).
    triton.bind(
        model_name="AddSub",
        infer_func=_add_sub,
        inputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        outputs=[
            Tensor(name="add", dtype=np.float32, shape=(-1,)),
            Tensor(name="sub", dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128),
    )
    logger.info("Blocks script while serving model")
    triton.serve()
