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
"""Very simple example with python identity operation."""
import logging

import numpy as np

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

logger = logging.getLogger("examples.identity_python.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


def _infer_raw_fn(inputs):  # noqa: N803
    return [
        {
            "OUTPUT_1": request["INPUT_1"],
            "OUTPUT_2": request["INPUT_2"],
        }
        for request in inputs
    ]


@batch
def _infer_fn(**inputs):  # noqa: N803
    return {
        "OUTPUT_1": inputs["INPUT_1"],
        "OUTPUT_2": inputs["INPUT_2"],
    }


with Triton() as triton:
    logger.info("Loading Identity model.")
    triton.bind(
        model_name="Identity",
        infer_func=_infer_fn,
        inputs=[
            Tensor(dtype=np.float64, shape=(-1,)),
            Tensor(dtype=object, shape=(1,)),
        ],
        outputs=[
            Tensor(dtype=np.float64, shape=(-1,)),
            Tensor(dtype=object, shape=(1,)),
        ],
        config=ModelConfig(max_batch_size=128),
    )
    logger.info("Serving inference")
    triton.serve()
