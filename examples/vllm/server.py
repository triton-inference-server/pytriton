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
"""Example with vllm engine."""

# Adapted from https://github.com/vllm-project/vllm/blob/v0.2.3/vllm/entrypoints/api_server.py

import argparse
import logging
from typing import AsyncGenerator, Dict, List

import numpy as np

from pytriton.model_config import ModelConfig, Tensor
from pytriton.proxy.types import Request
from pytriton.triton import Triton, TritonConfig
from vllm.engine.arg_utils import AsyncEngineArgs  # pytype: disable=import-error
from vllm.engine.async_llm_engine import AsyncLLMEngine  # pytype: disable=import-error
from vllm.sampling_params import SamplingParams  # pytype: disable=import-error
from vllm.utils import random_uuid  # pytype: disable=import-error

LOGGER = logging.getLogger("examples.vllm.server")


engine = None


async def _generate_for_request(request: Request) -> AsyncGenerator[Dict[str, np.ndarray], None]:
    """Generate completion for the request."""

    prompt = request.data["prompt"].tolist()[0].decode("utf-8")
    stream = request.data.get("stream", False)

    sampling_params = {name: value.item() for name, value in request.data.items() if name not in ("prompt", "stream")}
    sampling_params = SamplingParams(**sampling_params)
    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)

    if stream:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [prompt + output.text for output in request_output.outputs]
            yield {"text": np.char.encode(np.array(text_outputs)[None, ...], "utf-8")}
    else:
        final_output = None

        async for request_output in results_generator:
            final_output = request_output

        if final_output is None:
            raise ValueError("No output generated")

        prompt = final_output.prompt
        text_outputs = [prompt + output.text for output in final_output.outputs]

        yield {"text": np.char.encode(np.array(text_outputs)[None, ...], "utf-8")}


async def generate_fn(requests: List[Request]) -> AsyncGenerator[List[Dict[str, np.ndarray]], None]:
    assert len(requests) == 1, "expected single request because triton batching is disabled"
    request = requests[0]
    async for response in _generate_for_request(request):
        yield [response]  # ensure that the response is a list of responses of len 1, same as requests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--verbose", action="store_true")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    with Triton(config=TritonConfig(http_address=args.host, http_port=args.port)) as triton:
        triton.bind(
            model_name=args.model.replace("-", "_").replace("/", "_"),
            infer_func=generate_fn,
            inputs=[
                Tensor(name="prompt", dtype=bytes, shape=(1,)),
                Tensor(name="n", dtype=np.int32, shape=(1,), optional=True),
                Tensor(name="best_of", dtype=np.int32, shape=(1,), optional=True),
                Tensor(name="use_beam_search", dtype=np.bool_, shape=(1,), optional=True),
                Tensor(name="temperature", dtype=np.float32, shape=(1,), optional=True),
                Tensor(name="top_p", dtype=np.float32, shape=(1,), optional=True),
                Tensor(name="max_tokens", dtype=np.int32, shape=(1,), optional=True),
                Tensor(name="ignore_eos", dtype=np.bool_, shape=(1,), optional=True),
                Tensor(name="stream", dtype=np.bool_, shape=(1,), optional=True),
            ],
            outputs=[Tensor(name="text", dtype=bytes, shape=(-1, 1))],
            config=ModelConfig(batching=False, max_batch_size=128, decoupled=True),
            strict=True,
        )
        triton.serve()
