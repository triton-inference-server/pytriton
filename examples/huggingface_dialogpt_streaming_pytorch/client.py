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
"""Client for HF microsoft/DialoGPT sample server."""
import argparse
import logging

import numpy as np

from pytriton.client import DecoupledModelClient

_LOGGER = logging.getLogger("examples.huggingface_dialogpt_pytroch_streaming.client")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="grpc://localhost:8001",
        help=(
            "Url to Triton server (ex. grpc://localhost:8001)." "GRPC protocol is only supported with decoupled models"
        ),
        required=False,
    )
    parser.add_argument("--model-name", default="DialoGPT-small", help="Name of the model", required=False)
    parser.add_argument(
        "--init-timeout-s",
        type=float,
        default=600.0,
        help="Server and model ready state timeout in seconds",
        required=False,
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=8,
        help="Number of requests per client.",
        required=False,
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    if args.interactive:
        first_input = (input("TYPE AND PRESS ENTER:")).encode("utf-8")
    else:
        first_input = b"Does money buy happines?"

    chat_history_items = [first_input]  # initial prompt
    with DecoupledModelClient(args.url, args.model_name, init_timeout_s=args.init_timeout_s) as client:
        chat_history = b""
        print("(0) >", chat_history_items[0].decode("utf-8"))  # noqa: T201

        def idx_generator():
            if args.interactive:
                i = 1
                while True:
                    yield i
                    i += 1
            else:
                yield from range(1, args.iterations)

        chat_history = chat_history_items[0]
        for idx in idx_generator():

            if idx > 0:
                print(f"({idx}) > ", end="", flush=True)  # noqa: T201
            for partial_result_dict in client.infer_sample(
                new_inputs=np.array(chat_history_items[-1:]), chat_history=np.array([chat_history])
            ):

                response_tokens = partial_result_dict["response"].tolist()  # pytype: disable=unsupported-operands
                chat_history_items.extend(response_tokens)  # noqa: T201

                response_tokens = "".join(token.decode("utf-8") for token in response_tokens)
                print(response_tokens, end="", flush=True)  # noqa: T201
            print("")  # noqa: T201
            if args.interactive:
                next_input = (input("TYPE AND PRESS ENTER:")).encode("utf-8")
                chat_history_items.append(next_input)


if __name__ == "__main__":
    main()
