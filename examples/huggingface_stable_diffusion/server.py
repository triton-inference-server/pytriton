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
"""Server for Stable Diffusion 1.5."""
import argparse
import base64
import io
import logging

import numpy as np
import torch  # pytype: disable=import-error
from diffusers import StableDiffusionPipeline  # pytype: disable=import-error

from pytriton.decorators import batch, first_value, group_by_values
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

LOGGER = logging.getLogger("examples.huggingface_stable_diffusion.server")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_FORMAT = "JPEG"

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to(DEVICE)


def _encode_image_to_base64(image):
    raw_bytes = io.BytesIO()
    image.save(raw_bytes, IMAGE_FORMAT)
    raw_bytes.seek(0)  # return to the start of the buffer
    return base64.b64encode(raw_bytes.read())


@batch
@group_by_values("img_size")
@first_value("img_size")
def _infer_fn(
    prompt: np.ndarray,
    img_size: np.int64,
):
    prompts = [np.char.decode(p.astype("bytes"), "utf-8").item() for p in prompt]
    LOGGER.debug(f"Prompts: {prompts}")
    LOGGER.debug(f"Image Size: {img_size}x{img_size}")

    outputs = []
    for idx, image in enumerate(
        pipe(
            prompt=prompts,
            height=img_size,
            width=img_size,
        ).images
    ):
        raw_data = _encode_image_to_base64(image)
        outputs.append([raw_data])
        LOGGER.debug(f"Generated result for prompt `{prompts[idx]}` with size {len(raw_data)}")

    LOGGER.debug(f"Prepared batch response of size: {len(outputs)}")
    return {"image": np.array(outputs)}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging in debug mode.",
    )
    return parser.parse_args()


def main():
    """Initialize server with model."""
    args = _parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    log_verbose = 1 if args.verbose else 0
    config = TritonConfig(exit_on_error=True, log_verbose=log_verbose)

    with Triton(config=config) as triton:
        LOGGER.info("Loading the pipeline")
        triton.bind(
            model_name="StableDiffusion_1_5",
            infer_func=_infer_fn,
            inputs=[
                Tensor(name="prompt", dtype=np.bytes_, shape=(1,)),
                Tensor(name="img_size", dtype=np.int64, shape=(1,)),
            ],
            outputs=[
                Tensor(name="image", dtype=np.bytes_, shape=(1,)),
            ],
            config=ModelConfig(
                max_batch_size=4,
                batcher=DynamicBatcher(
                    max_queue_delay_microseconds=100,
                ),
            ),
        )
        triton.serve()


if __name__ == "__main__":
    main()
