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
"""Simple classifier example based on Hugging Face JAX BERT model."""

import logging

import numpy as np
from transformers import BertTokenizer, FlaxBertModel  # pytype: disable=import-error

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

logger = logging.getLogger("examples.huggingface_bert_jax.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = FlaxBertModel.from_pretrained("bert-base-uncased")


@batch
def _infer_fn(**inputs: np.ndarray):
    (sequence_batch,) = inputs.values()

    # need to convert dtype=object to bytes first
    # end decode unicode bytes
    sequence_batch = np.char.decode(sequence_batch.astype("bytes"), "utf-8")

    last_hidden_states = []
    for sequence_item in sequence_batch:
        tokenized_sequence = tokenizer(sequence_item.item(), return_tensors="jax")
        results = model(**tokenized_sequence)
        last_hidden_states.append(results.last_hidden_state)
    last_hidden_states = np.array(last_hidden_states, dtype=np.float32)
    return [last_hidden_states]


with Triton() as triton:
    logger.info("Loading BERT model.")
    triton.bind(
        model_name="BERT",
        infer_func=_infer_fn,
        inputs=[
            Tensor(name="sequence", dtype=np.bytes_, shape=(1,)),
        ],
        outputs=[
            Tensor(
                name="last_hidden_state",
                dtype=np.float32,
                shape=(-1, -1, -1),
            ),
        ],
        config=ModelConfig(max_batch_size=16),
        strict=True,
    )
    logger.info("Serving inference")
    triton.serve()
