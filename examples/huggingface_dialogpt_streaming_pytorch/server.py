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
"""Simple classifier example based on HF microsoft/DialoGPT model."""

import argparse
import concurrent
import logging
import pathlib
import queue
import typing

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer  # pytype: disable=import-error

from pytriton.decorators import batch  # pytype: disable=import-error
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

logger = logging.getLogger("examples.huggingface_dialogpt_pytroch_streaming.server")


class StreamingBot:
    def __init__(self, model_name: str, *, max_length: int = 1000, timeout_s: typing.Optional[float] = None) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self._timeout_s = timeout_s
        self._max_length = max_length

        self.streamer = TextIteratorStreamer(
            self.tokenizer, timeout=self._timeout_s, skip_prompt=True, skip_special_tokens=True
        )
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="streaming_bot")

    @batch
    def __call__(self, new_inputs, chat_history=None) -> typing.Any:
        inputs_kwargs = self._prepare_inputs(new_inputs, chat_history)  # inputs_ids + attention_mask named args
        generate_kwargs = dict(
            **inputs_kwargs,
            streamer=self.streamer,
            max_length=self._max_length,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generate_future = self._executor.submit(self.model.generate, **generate_kwargs)

        try:
            for token in self.streamer:
                yield {
                    "response": np.char.encode([token], "utf-8")[np.newaxis, ...]
                }  # add batch dimension to match declared signature
        except queue.Empty:
            generate_future.cancel()
            raise TimeoutError(f"Timeout occurred during model generation (timeout_s={self._timeout_s}).") from None

        generate_future.result()  # raise exception if any occurred in model.generation method

    def _prepare_inputs(self, new_inputs, chat_history):
        new_inputs = np.char.decode(new_inputs.astype("bytes"), "utf-8")
        bot_inputs = np.char.add(new_inputs, self.tokenizer.eos_token)

        if chat_history:
            chat_history = np.char.decode(chat_history.astype("bytes"), "utf-8")
            bot_inputs = np.char.add(chat_history, bot_inputs)

        bot_inputs = bot_inputs[..., 0].tolist()  # reduce to 1D list
        model_kwargs = self.tokenizer.batch_encode_plus(bot_inputs, return_tensors="pt", padding=True)
        return model_kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-batch-size", type=int, default=8, help="Batch size of request.", required=False)
    parser.add_argument("--model-name", default="microsoft/DialoGPT-small", help="Name of the model", required=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    verbose_level = int(args.verbose) * 3
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    streaming_bot = StreamingBot(args.model_name, timeout_s=10.0)

    with Triton(config=TritonConfig(log_verbose=verbose_level)) as triton:
        triton.bind(
            model_name=pathlib.Path(args.model_name).stem,
            infer_func=streaming_bot,
            inputs=[
                Tensor(name="new_inputs", dtype=bytes, shape=(1,)),
                Tensor(name="chat_history", optional=True, dtype=bytes, shape=(1,)),
            ],
            outputs=[
                Tensor(name="response", dtype=bytes, shape=(1,)),
            ],
            config=ModelConfig(decoupled=True),
            strict=True,
        )
        triton.serve()


if __name__ == "__main__":
    main()
