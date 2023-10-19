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
"""Starting PyTriton in remote mode and binding model calculating element wise power."""
import logging

import numpy as np

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import RemoteTriton

logger = logging.getLogger("examples.simple_python_remote_mode.server_remote_power")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


@batch
def _power(**inputs):
    a_batch, b_batch = inputs.values()
    dot_result = np.power(a_batch, b_batch)
    return {"power": dot_result}


with RemoteTriton(url="localhost") as remote_triton:
    remote_triton.bind(
        model_name="Power",
        infer_func=_power,
        inputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        outputs=[
            Tensor(name="power", dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128),
    )

    remote_triton.serve()
