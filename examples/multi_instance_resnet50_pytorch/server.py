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
import argparse
import logging
from typing import Any, List

import numpy as np
import torch  # pytype: disable=import-error
from transformers import ResNetForImageClassification  # pytype: disable=import-error

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

logger = logging.getLogger("examples.multi_instance_resnet50_pytorch.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

DEVICE = "cuda:0"


class _InferFuncWrapper:
    def __init__(self, model: Any, device: str):
        self._model = model
        self._device = device

    @batch
    def __call__(self, image: np.ndarray):
        logger.debug(f"Image data: {image.shape} ({image.size})")

        image = torch.from_numpy(image).to(self._device)
        with torch.inference_mode():
            logits = self._model(pixel_values=image).logits
            logits = logits.to("cpu")

        labels = []
        for logit in logits:
            predicted_label = logit.argmax(-1).item()
            label = np.char.encode(self._model.config.id2label[predicted_label], "utf-8")
            labels.append([label])

        return {"label": np.array(labels)}


def _infer_function_factory(devices: List[str]):
    infer_funcs = []
    for device in devices:
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        model = model.half().to(device).eval()
        infer_funcs.append(_InferFuncWrapper(model=model, device=device))

    return infer_funcs


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
        "--number-of-instances",
        type=int,
        default=2,
        help="Batch size of request.",
        required=False,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    devices = [DEVICE] * args.number_of_instances
    with Triton() as triton:
        logger.info(f"Loading ResNet50 PyTorch model on devices: {devices}")
        triton.bind(
            model_name="ResNet50",
            infer_func=_infer_function_factory(devices),
            inputs=[
                Tensor(
                    name="image",
                    dtype=np.float16,
                    shape=(3, 224, 224),
                ),
            ],
            outputs=[
                Tensor(name="label", dtype=bytes, shape=(1,)),
            ],
            config=ModelConfig(
                max_batch_size=args.max_batch_size,
            ),
        )
        logger.info("Serving model")
        triton.serve()


if __name__ == "__main__":
    main()
