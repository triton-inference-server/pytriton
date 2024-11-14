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

# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.2.1/vllm/entrypoints/api_server.py

import argparse

import numpy as np
from vllm.engine.arg_utils import AsyncEngineArgs  # pytype: disable=import-error
from vllm.engine.async_llm_engine import AsyncLLMEngine  # pytype: disable=import-error
from vllm.sampling_params import SamplingParams  # pytype: disable=import-error
from vllm.utils import random_uuid  # pytype: disable=import-error

from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
engine = None


async def _generate_for_request(request):
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    prompt = request.data.get("prompt").tolist()[0].decode("utf-8")
    stream = request.parameters.get("stream", False)

    sampling_params = {name: value.item() for name, value in request.data.items() if name not in ("prompt", "stream")}
    sampling_params = SamplingParams(**sampling_params)
    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)

    if stream:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [prompt + output.text for output in request_output.outputs]
            yield {"text": np.array(text_outputs, dtype=object)}
    else:
        final_output = None

        async for request_output in results_generator:
            # if await request.is_disconnected():
            #     # Abort the request if the client disconnects.
            #     await engine.abort(request_id)
            #     return Response(status_code=499)
            final_output = request_output

        assert final_output is not None

        prompt = final_output.prompt
        text_outputs = [prompt + output.text for output in final_output.outputs]

        yield {"text": np.char.encode(np.array(text_outputs)[None, ...], "utf-8")}


async def generate_fn(requests):
    assert len(requests) == 1
    async for response in _generate_for_request(requests[0]):
        yield [response]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    with Triton(config=TritonConfig(http_address=args.host, http_port=args.port)) as triton:
        triton.bind(
            model_name=args.model.replace("-", "_"),
            infer_func=generate_fn,
            inputs=[
                Tensor(name="prompt", dtype=bytes, shape=(1,)),
                Tensor(name="n", dtype=np.int32, shape=(1,)),
                Tensor(name="best_of", dtype=np.int32, shape=(1,)),
                Tensor(name="temperature", dtype=np.float32, shape=(1,)),
                Tensor(name="top_p", dtype=np.float32, shape=(1,)),
                Tensor(name="max_tokens", dtype=np.int32, shape=(1,)),
                Tensor(name="ignore_eos", dtype=np.bool_, shape=(1,)),
                Tensor(name="stream", dtype=np.bool_, shape=(1,)),
            ],
            outputs=[Tensor(name="text", dtype=bytes, shape=(-1, 1))],
            config=ModelConfig(batching=False, max_batch_size=128, decoupled=True),
            strict=True,
        )
        triton.serve()
