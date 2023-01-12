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
"""Simple classifier example based on Hugging Face Pytorch BART model."""
import logging

import numpy as np
from transformers import pipeline  # pytype: disable=import-error

from pytriton.decorators import sample
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

CLASSIFIER = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)


@sample
def _infer_fn(sequence: np.ndarray, labels: np.ndarray):
    sequence = np.char.decode(sequence.astype("bytes"), "utf-8")  # need to convert dtype=object to bytes first
    labels = np.char.decode(labels.astype("bytes"), "utf-8")
    classification_result = CLASSIFIER(sequence.item(), labels.tolist())
    scores_batch = np.array(classification_result["scores"], dtype=np.float32)
    return {"scores": scores_batch}


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
logger = logging.getLogger("examples.huggingface_bart_pytorch.server")

with Triton() as triton:
    logger.info("Loading BART model.")
    triton.bind(
        model_name="BART",
        infer_func=_infer_fn,
        inputs=[
            Tensor(name="sequence", dtype=bytes, shape=(1,)),
            Tensor(name="labels", dtype=bytes, shape=(-1,)),
        ],
        outputs=[
            Tensor(name="scores", dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(batching=False),
    )
    logger.info("Serving inference")
    triton.serve()
