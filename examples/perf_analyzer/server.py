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

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

logger = logging.getLogger("examples.perf_analyzer.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)


@batch
def _infer_fn(**inputs: np.ndarray):
    sequences_batch, labels_batch = inputs.values()

    # need to convert dtype=object to bytes first
    # end decode unicode bytes
    sequences_batch = np.char.decode(sequences_batch.astype("bytes"), "utf-8")
    labels_batch = np.char.decode(labels_batch.astype("bytes"), "utf-8")

    scores = []
    for sequence, labels in zip(sequences_batch, labels_batch):
        classification_result = classifier(sequence.item(), labels.tolist())
        scores.append(classification_result["scores"])
    scores_batch = np.array(scores, dtype=np.float32)
    return {"scores": scores_batch}


with Triton() as triton:
    logger.info("Loading BART model.")
    triton.bind(
        model_name="BART",
        infer_func=_infer_fn,
        inputs=[
            Tensor(name="sequence", dtype=np.bytes_, shape=(1,)),
            Tensor(name="labels", dtype=np.bytes_, shape=(-1,)),
        ],
        outputs=[
            Tensor(name="scores", dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=8),
    )
    logger.info("Serving inference")
    triton.serve()
