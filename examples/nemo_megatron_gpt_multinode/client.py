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
import typing

import numpy as np

from pytriton.client import ModelClient

_AVAILABLE_TASKS = ["sentiment", "intent_and_slot", "text_generation"]
_TASK_SEP = "|"


def _parse_prompts(prompts_list) -> typing.List[typing.Tuple[str, str]]:
    """
    Parse prompts in the format of '[<task_name>:]<prompt>'.
    Available tasks: {', '.join(_AVAILABLE_TASKS)}. If you don't specify a task name, the model will default to text generation.
    """

    def _parse_prompt(prompt_str: str) -> typing.Tuple[str, str]:
        if _TASK_SEP in prompt_str:
            task_name, value = prompt_str.split(_TASK_SEP, 1)
            task_name = task_name.strip().lower()
        else:
            task_name = "text_generation"
            value = prompt_str.strip()
        return task_name, value

    return [_parse_prompt(prompt_str) for prompt_str in prompts_list]


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
        default=[
            "Q: How are you?",
            "Q: How big is the universe?",
            f"sentiment{_TASK_SEP}It estimates the operating profit to further improve from the third quarter.",
            f"intent_and_slot{_TASK_SEP}What is the weather like today?",
        ],
        nargs="+",
        help=(
            f"Prompts should be in the format of '[<task_name>{_TASK_SEP}]<prompt>'. "
            f"Available tasks: {', '.join(_AVAILABLE_TASKS)}. "
            "If you don't specify a task name, the model will default to text generation."
        ),
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--end-strings",
        nargs="+",
        default=["\n"],
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
    logger = logging.getLogger("nemo.client")

    tasks_and_prompts = _parse_prompts(args.prompts)
    tasks, prompts = tuple(zip(*tasks_and_prompts))
    if not all(task in _AVAILABLE_TASKS for task in tasks):
        raise ValueError(f"Unknown tasks: {set(tasks) - set(_AVAILABLE_TASKS)}")

    batch_size = len(args.prompts)

    def _str_list2numpy(str_list: typing.List[str]) -> np.ndarray:
        str_ndarray = np.array(str_list)[..., np.newaxis]
        return np.char.encode(str_ndarray, "utf-8")

    tasks = _str_list2numpy(tasks)
    prompts = _str_list2numpy(prompts)

    def _param(dtype, value):
        if bool(value):
            return np.ones((batch_size, 1), dtype=dtype) * value
        else:
            return np.zeros((batch_size, 1), dtype=dtype)

    logger.info("================================")
    logger.info("Preparing the client")
    with ModelClient(args.url, "GPT", init_timeout_s=args.init_timeout_s) as client:
        logger.info("================================")
        logger.info("Sent batch for inference:")

        result_dict = client.infer_batch(
            tasks=tasks,
            prompts=prompts,
            min_length=_param(np.int32, 20),
            max_length=_param(np.int32, args.output_len),
            use_greedy=_param(np.bool_, True),
            temperature=_param(np.float32, 1.0),
            top_k=_param(np.int32, 0),
            top_p=_param(np.float32, 1.0),
            repetition_penalty=_param(np.float32, 1.0),
            add_BOS=_param(np.bool_, True),
            all_probs=_param(np.bool_, False),
            compute_logprob=_param(np.bool_, False),
            end_strings=np.tile(np.char.encode(np.array(args.end_strings)[np.newaxis, np.newaxis, ...], "utf-8"), (batch_size, 1, 1)),
            #end_strings=np.tile(np.char.encode(np.array(args.end_strings)[np.newaxis, ...], "utf-8"), (batch_size, 1)),
        )

    sentences = np.char.decode(result_dict["sentences"].astype("bytes"), "utf-8")
    sentences = np.squeeze(sentences, axis=-1)
    for sentence in sentences:
        logger.info("================================")
        logger.info(sentence)


if __name__ == "__main__":
    main()
