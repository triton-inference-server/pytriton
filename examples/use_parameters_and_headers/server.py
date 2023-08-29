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
"""Server with simple python model performing adding and subtract operation using custom headers and parameters."""
import logging

import numpy as np

from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

logger = logging.getLogger("examples.use_parameters_and_headers.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


def _infer_with_params_and_headers(requests):
    responses = []
    for req in requests:
        a_batch, b_batch = req.values()
        scaled_add_batch = (a_batch + b_batch) / float(req.parameters["header_divisor"])
        scaled_sub_batch = (a_batch - b_batch) * float(req.parameters["parameter_multiplier"])
        responses.append({"scaled_add": scaled_add_batch, "scaled_sub": scaled_sub_batch})
    return responses


with Triton(config=TritonConfig(http_header_forward_pattern="header.*")) as triton:
    logger.info("Loading the model using parameters and headers")
    triton.bind(
        model_name="ParamsAndHeaders",
        infer_func=_infer_with_params_and_headers,
        inputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        outputs=[
            Tensor(name="scaled_add", dtype=np.float32, shape=(-1,)),
            Tensor(name="scaled_sub", dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128),
    )
    logger.info("Serving model")
    triton.serve()
