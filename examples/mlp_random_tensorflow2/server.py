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
"""Example with random MLP implemented with TF2 framework."""
import logging

import numpy as np
import tensorflow as tf  # pytype: disable=import-error

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

logger = logging.getLogger("examples.mlp_random_tensorflow2.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


def _get_model():  # Load model into Triton Inference Server

    input_layer = tf.keras.layers.Input((224, 224, 3))
    layer_output = tf.keras.layers.Lambda(lambda x: x)(input_layer)
    model_output = tf.keras.layers.Lambda(lambda x: x)(layer_output)
    model = tf.keras.Model(input_layer, model_output)
    return model


MODEL = _get_model()


@batch
def _infer_fn(image):
    images_batch_tensor = tf.convert_to_tensor(image)
    output1_batch = MODEL.predict(images_batch_tensor)
    return [output1_batch]


with Triton() as triton:
    logger.info("Loading MLP model.")
    triton.bind(
        model_name="MLP",
        infer_func=_infer_fn,
        inputs=[
            Tensor(name="image", dtype=np.float32, shape=(224, 224, 3)),
        ],
        outputs=[
            Tensor(name="output", dtype=np.float32, shape=(224, 224, 3)),
        ],
        config=ModelConfig(max_batch_size=16),
    )
    logger.info("Serving inference")
    triton.serve()
