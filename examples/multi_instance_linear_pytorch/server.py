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
import logging
from typing import List

import numpy as np
import torch  # pytype: disable=import-error

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

logger = logging.getLogger("examples.multi_instance_linear_pytorch.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

DEVICES = ["cuda", "cpu"]


class _InferFuncWrapper:
    def __init__(self, model: torch.nn.Module, device: str):
        self._model = model
        self._device = device

    @batch
    def __call__(self, **inputs):
        logger.info(f"Instance on {self._device}.")
        (input1_batch,) = inputs.values()
        input1_batch_tensor = torch.from_numpy(input1_batch).to(self._device)
        output1_batch_tensor = self._model(input1_batch_tensor)
        output1_batch = output1_batch_tensor.cpu().detach().numpy()
        return [output1_batch]


def _infer_function_factory(devices: List[str]):
    infer_funcs = []
    for device in devices:
        model = torch.nn.Linear(20, 30).to(device).eval()
        infer_funcs.append(_InferFuncWrapper(model=model, device=device))

    return infer_funcs


with Triton() as triton:

    logger.info("Loading Linear PyTorch model on devices: {devices}")
    triton.bind(
        model_name="Linear",
        infer_func=_infer_function_factory(DEVICES),
        inputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        outputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128),
        strict=True,
    )
    logger.info("Serving model")
    triton.serve()
