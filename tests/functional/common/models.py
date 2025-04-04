# Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
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
import dataclasses
import enum
from typing import Callable, Optional, Sequence

import numpy as np

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor


class Framework(enum.Enum):
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"


@dataclasses.dataclass(frozen=True)
class TestModelSpec:
    name: str
    framework: Optional[Framework]
    create_infer_fn: Callable[..., Callable]
    inputs: Sequence[Tensor]
    outputs: Sequence[Tensor]
    model_config: ModelConfig


def _create_add_sub_fn() -> Callable:
    @batch
    def _add_sub(**inputs):
        a_batch, b_batch = inputs.values()
        add_batch = a_batch + b_batch
        sub_batch = a_batch - b_batch
        return {"add": add_batch, "sub": sub_batch}

    return _add_sub


ADD_SUB_PYTHON_MODEL = TestModelSpec(
    name="AddSub",
    framework=None,
    create_infer_fn=_create_add_sub_fn,
    inputs=(
        Tensor(dtype=np.float32, shape=(-1,)),
        Tensor(dtype=np.float32, shape=(-1,)),
    ),
    outputs=(
        Tensor(name="add", dtype=np.float32, shape=(-1,)),
        Tensor(name="sub", dtype=np.float32, shape=(-1,)),
    ),
    model_config=ModelConfig(max_batch_size=128),
)


def _create_identity_fn() -> Callable:
    @batch
    def _identity(**inputs):
        (a_batch,) = inputs.values()
        return {"identity": a_batch}

    return _identity


IDENTITY_PYTHON_MODEL = TestModelSpec(
    name="Identity",
    framework=None,
    create_infer_fn=_create_identity_fn,
    inputs=(Tensor(dtype=np.float32, shape=(-1,)),),
    outputs=(Tensor(name="identity", dtype=np.float32, shape=(-1,)),),
    model_config=ModelConfig(max_batch_size=128),
)


def _create_tfhub_tensorflow_efficientdet_lite0_detection_fn() -> Callable:
    import tensorflow_hub as hub  # pytype: disable=import-error

    detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1")

    @batch
    def _tfhub_tensorflow_efficientdet_lite0_detection(image: np.ndarray):
        boxes, scores, classes, num_detections = detector(image)
        return {
            "boxes": boxes.numpy(),
            "scores": scores.numpy(),
            "classes": classes.numpy(),
            "num_detections": num_detections.numpy(),
        }

    return _tfhub_tensorflow_efficientdet_lite0_detection


EFFICIENTDET_DETECTION_TF_MODEL = TestModelSpec(
    name="EfficientDetDetector",
    framework=Framework.TENSORFLOW,
    create_infer_fn=_create_tfhub_tensorflow_efficientdet_lite0_detection_fn,
    inputs=(Tensor(name="image", dtype=np.uint8, shape=(-1, -1, 3)),),
    outputs=(
        Tensor(
            name="boxes",
            dtype=np.float32,
            shape=(-1, 4),
        ),
        Tensor(
            name="scores",
            dtype=np.float32,
            shape=(-1,),
        ),
        Tensor(
            name="classes",
            dtype=np.int32,
            shape=(-1,),
        ),
        Tensor(name="num_detections", dtype=np.int32, shape=(1,)),
    ),
    model_config=ModelConfig(max_batch_size=128),
)


MODELS_CATALOGUE = [
    ADD_SUB_PYTHON_MODEL,
    IDENTITY_PYTHON_MODEL,
    EFFICIENTDET_DETECTION_TF_MODEL,
]
