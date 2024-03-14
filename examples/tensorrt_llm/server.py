#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""Example use of TensorRT-LLM with PyTriton."""


import argparse
import contextlib
import json
import logging
import pathlib
import typing

import numpy as np

from pytriton.model_config import ModelConfig, Tensor
from pytriton.proxy.types import Request
from pytriton.triton import Triton, TritonConfig

LOGGER = logging.getLogger("examples.tensorrt_llm.server")


class GptDeployment:
    def __init__(self, engine_dir: pathlib.Path, tokenizer: typing.Union[str, pathlib.Path]):
        """Initialize a new GPT deployment.

        Args:
            engine_dir: The path to the TensorRT-LLM engine directory.
            tokenizer: The path to the tokenizer or the name of the HF tokenizer.
        """
        from tensorrt_llm.executor import GenerationExecutor  # pytype: disable=import-error
        from tensorrt_llm.hlapi.tokenizer import TransformersTokenizer  # pytype: disable=import-error

        tokenizer = TransformersTokenizer.from_pretrained(
            tokenizer,
            legacy=False,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=True,
            use_fast=True,
        )

        engine_config_path = engine_dir / "config.json"
        with engine_config_path.open("r") as engine_config_file:
            engine_config = json.load(engine_config_file)
            tp_size = engine_config["pretrained_config"]["mapping"]["tp_size"]
            pp_size = engine_config["pretrained_config"]["mapping"]["pp_size"]
            assert tp_size == 1, "Example doesn't support Tensor parallelism."
            assert pp_size == 1, "Example doesn't support Pipeline parallelism."

        self._engine = GenerationExecutor(engine_dir=engine_dir, tokenizer=tokenizer)

    async def infer_fn(
        self, requests: typing.List[Request]
    ) -> typing.AsyncGenerator[typing.List[typing.Dict[str, np.ndarray]], None]:
        # unpack the request (single prompt in single item requests list)
        request = {key: value[0] for key, value in requests[0].items()}

        async for result in self._engine.generate_async(
            request["text_input"].decode("utf-8"),
            streaming=request["stream"].item(),
            max_new_tokens=request["max_tokens"].item(),
        ):
            text_output = result.text
            # wrap in array to match (1, ) shape
            result_dict = {"text_output": np.char.encode(np.array([text_output]), "utf-8")}
            yield [result_dict]  # wrap in list to match single item requests list

    @property
    def inputs(self):
        return [
            Tensor(name="text_input", dtype=bytes, shape=(1,)),
            Tensor(name="max_tokens", dtype=np.int32, shape=(1,)),
            Tensor(name="stream", dtype=np.bool_, shape=(1,)),
        ]

    @property
    def outputs(self):
        return [
            Tensor(name="text_output", dtype=bytes, shape=(1,)),
        ]

    def close(self):
        if self._engine is not None and hasattr(self._engine, "shutdown"):
            self._engine.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model deployment on Triton")
    parser.add_argument(
        "--engine-dir",
        type=pathlib.Path,
        required=True,
        help="Path to the TensorRT-LLM engine directory",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to the directory with tokenizer files or the name of the HF tokenizer",
    )
    parser.add_argument("--host", type=str, default=None, help="Host to bind the server to")
    parser.add_argument("--http-port", type=int, default=8000, help="Port to bind the HTTP server to")
    parser.add_argument("--grpc-port", type=int, default=8001, help="Port to bind the gRPC server to")
    parser.add_argument("--metrics-port", type=int, default=8002, help="Port to bind the metrics server to")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    with contextlib.closing(GptDeployment(args.engine_dir, args.tokenizer)) as gpt_deploy:
        triton_config = TritonConfig(
            http_address=args.host,
            http_port=args.http_port,
            grpc_address=args.host,
            grpc_port=args.grpc_port,
            metrics_address=args.host,
            metrics_port=args.metrics_port,
        )

        with Triton(config=triton_config) as triton:
            triton.bind(
                model_name=args.model_name,
                infer_func=gpt_deploy.infer_fn,
                inputs=gpt_deploy.inputs,
                outputs=gpt_deploy.outputs,
                config=ModelConfig(batching=False, decoupled=True),
                strict=True,
            )
            triton.serve()


if __name__ == "__main__":
    main()
