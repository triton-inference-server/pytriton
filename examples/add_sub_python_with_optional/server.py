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
"""Server with simple python model performing adding and subtract operation with optional 'w' and 't' params."""
import logging

import numpy as np

from pytriton.decorators import batch, fill_optionals, first_value
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

LOGGER = logging.getLogger("examples.add_sub_python_with_optional.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


@fill_optionals(t=np.array([2.0], dtype=np.float32))
@batch
@first_value("w")
def _add_sub(a, b, t, **inputs):
    w = 1 if "w" not in inputs else inputs["w"]
    add_batch = a * w + b + t
    sub_batch = a * w - b + t
    return {"add": add_batch, "sub": sub_batch}


def main():
    with Triton() as triton:
        LOGGER.info("Loading AddSub model")
        triton.bind(
            model_name="AddSub",
            infer_func=_add_sub,
            inputs=[
                Tensor(dtype=np.float32, shape=(-1,), name="a"),
                Tensor(dtype=np.float32, shape=(-1,), name="b"),
                Tensor(dtype=np.float32, shape=(-1,), name="t", optional=True),
                Tensor(dtype=np.float32, shape=(-1,), name="w", optional=True),
            ],
            outputs=[
                Tensor(name="add", dtype=np.float32, shape=(-1,)),
                Tensor(name="sub", dtype=np.float32, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=128),
            strict=True,
        )
        LOGGER.info("Serving model")
        triton.serve()


if __name__ == "__main__":
    main()
