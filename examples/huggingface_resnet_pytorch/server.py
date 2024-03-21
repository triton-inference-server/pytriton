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
"""Simple classifier example based on Hugging Face Pytorch ResNet model."""

import argparse
import io
import logging

import numpy as np
import torch  # pytype: disable=import-error
from PIL import Image  # pytype: disable=import-error
from transformers import AutoFeatureExtractor, ResNetForImageClassification  # pytype: disable=import-error

from pytriton.decorators import batch
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton

logger = logging.getLogger("examples.huggingface_bart_pytorch.server")

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(DEVICE)


@batch
def _infer_fn(image: np.ndarray):
    logger.debug(f"Image data: {image.shape} ({image.size})")
    images = []
    for img in image:
        img = Image.open(io.BytesIO(img.tobytes()))
        images.append(img)

    inputs = feature_extractor(images, return_tensors="pt")
    for name, value in inputs.items():
        inputs[name] = value.to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        logits = logits.to("cpu")

    labels = []
    for logit in logits:
        predicted_label = logit.argmax(-1).item()
        label = np.char.encode(model.config.id2label[predicted_label], "utf-8")
        labels.append([label])

    return {"label": np.array(labels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
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
        logger.info("Loading ResNet model.")
        triton.bind(
            model_name="ResNet",
            infer_func=_infer_fn,
            inputs=[
                Tensor(name="image", dtype=np.uint8, shape=(-1,)),
            ],
            outputs=[
                Tensor(name="label", dtype=bytes, shape=(1,)),
            ],
            config=ModelConfig(
                max_batch_size=args.max_batch_size,
                batcher=DynamicBatcher(max_queue_delay_microseconds=5000),  # 5ms
            ),
            strict=True,
        )
        logger.info("Serving inference")
        triton.serve()


if __name__ == "__main__":
    main()
