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
"""Client for FasterTransformer GPT server."""
import argparse
import logging

import numpy as np

from pytriton.client import ModelClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="localhost",
        help=(
            "Url to Triton server (ex. grpc://localhost:8000)."
            "HTTP protocol with default port is used if parameter is not provided"
        ),
    )
    parser.add_argument("--model-name", default="gpt2", help="Name of the model")
    parser.add_argument(
        "--init-timeout-s",
        type=float,
        default=600.0,
        help="Server and model ready state timeout in seconds",
    )
    parser.add_argument(
        "--prompts",
        default=["Q: How are you?", "Q: How big is the universe?", "1 2 3 4 ", "a b c d "],
        nargs="+",
        help="Prompts to form request",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=30,
        help="Number of output tokens",
    )
    parser.add_argument("--beam-width", default=1, type=int, help="Beam search width")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    sequences = np.array(args.prompts)[..., np.newaxis]
    sequences = np.char.encode(sequences, "utf-8")
    batch_size = len(sequences)

    def _param(dtype, value):
        if bool(value):
            return np.ones((batch_size, 1), dtype=dtype) * value
        else:
            return np.zeros((batch_size, 1), dtype=dtype)

    with ModelClient(args.url, args.model_name, init_timeout_s=args.init_timeout_s) as client:
        result_dict = client.infer_batch(
            prompts=sequences,
            request_output_len=_param(np.uint32, args.output_len),
            beam_width=_param(np.uint32, args.beam_width),
        )

    def _convert_str_if_applicable(value):
        if isinstance(value, np.ndarray) and value.dtype == np.object_:
            value = np.char.decode(value.astype("bytes"), "utf-8")
            value = np.squeeze(value, axis=-1)
        return value

    result_dict = {key: _convert_str_if_applicable(value) for key, value in result_dict.items()}
    for sample_idx in range(batch_size):
        print(f"================[ {sample_idx} ]================")  # noqa
        print(f"prompt: {args.prompts[sample_idx]}")  # noqa
        for key, value in result_dict.items():
            print(f"{key}: {value[sample_idx].tolist()}")  # noqa


if __name__ == "__main__":
    main()
