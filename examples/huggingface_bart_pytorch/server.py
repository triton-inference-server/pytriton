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
import argparse
import logging

import numpy as np
from transformers import pipeline  # pytype: disable=import-error

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

logger = logging.getLogger("examples.huggingface_bart_pytorch.server")

CLASSIFIER = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

# Labels pre-cached on server side
LABELS = [
    "travel",
    "cooking",
    "dancing",
    "sport",
    "music",
    "entertainment",
    "festival",
    "movie",
    "literature",
]


@batch
def _infer_fn(sequence: np.ndarray):
    sequence = np.char.decode(sequence.astype("bytes"), "utf-8")  # need to convert dtype=object to bytes first
    sequence = sequence.tolist()

    classification_result = CLASSIFIER(sequence, LABELS)
    result_labels = []
    for result in classification_result:
        logger.debug(result)
        label = result["labels"][0]
        result_labels.append(label)

    return {"labels": np.char.encode(result_labels, "utf-8")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Batch size of request.",
        required=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    with Triton() as triton:
        logger.info("Loading BART model.")
        triton.bind(
            model_name="BART",
            infer_func=_infer_fn,
            inputs=[
                Tensor(name="sequence", dtype=bytes, shape=(1,)),
            ],
            outputs=[
                Tensor(name="labels", dtype=bytes, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=args.max_batch_size),
        )
        logger.info("Serving inference")
        triton.serve()


if __name__ == "__main__":
    main()
