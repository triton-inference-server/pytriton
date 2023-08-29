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
"""Example with single random Linear implemented with PyTorch framework."""
import logging

import numpy as np
import torch  # pytype: disable=import-error

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = torch.nn.Linear(20, 30).to(DEVICE).eval()


@batch
def _infer_fn(**inputs):
    (input1_batch,) = inputs.values()
    input1_batch_tensor = torch.from_numpy(input1_batch).to(DEVICE)
    output1_batch_tensor = MODEL(input1_batch_tensor)
    output1_batch = output1_batch_tensor.cpu().detach().numpy()
    return [output1_batch]


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
logger = logging.getLogger("examples.linear_random_pytorch.server")

with Triton() as triton:
    logger.info("Loading Linear model.")
    triton.bind(
        model_name="Linear",
        infer_func=_infer_fn,
        inputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        outputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128),
    )
    logger.info("Serving models")
    triton.serve()
