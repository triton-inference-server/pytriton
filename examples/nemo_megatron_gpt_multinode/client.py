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
"""Client for BART classifier sample server."""
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
    parser.add_argument(
        "--init-timeout-s",
        type=float,
        default=600.0,
        help="Server and model ready state timeout in seconds",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=30,
        help="Number of output tokens",
    )
    parser.add_argument(
        "--prompts",
        default=["Q: How are you?", "Q: How big is the universe?"],
        nargs="+",
        help="Prompts to form request",
    )

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

    with ModelClient(args.url, "GPT", init_timeout_s=args.init_timeout_s) as client:
        # parameters values taken from megatron_gpt_inference.yaml conf
        # here is another set of parameters https://github.com/NVIDIA/NeMo/blob/main/examples/nlp/language_modeling/megatron_gpt_eval.py#L119
        result_dict = client.infer_batch(
            sentences=sequences,
            tokens_to_generate=_param(np.int32, args.output_len),
            min_tokens_to_generate=_param(np.int32, 0),
            all_probs=_param(np.bool_, False),
            temperature=_param(np.float32, 1.0),
            add_BOS=_param(np.bool_, True),
            top_k=_param(np.int32, 0),
            top_p=_param(np.float32, 0.9),
            greedy=_param(np.bool_, False),
            repetition_penalty=_param(np.float32, 1.2),
        )

    sentences = np.char.decode(result_dict["sentences"].astype("bytes"), "utf-8")
    sentences = np.squeeze(sentences, axis=-1)
    for sentence in sentences:
        print("================================")  # noqa
        print(sentence)  # noqa


if __name__ == "__main__":
    main()
