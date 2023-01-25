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
from typing import Dict

import numpy as np
import torch  # pytype: disable=import-error
from utils import broadcast_inputs  # pytype: disable=import-error

from pytriton.decorators import batch
from pytriton.exceptions import PytritonUnrecoverableError
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig


class FasterTransformerTritonServer:
    _shared_tensors_dict = {"start_ids": None, "start_lengths": None}

    def __init__(self, model, tokenizer, *, model_name: str):
        self._model = model
        self._model_name = model_name

        # try to match signature from
        # https://github.com/triton-inference-server/fastertransformer_backend/blob/main/all_models/gpt/ensemble/config.pbtxt
        inputs = (
            Tensor(name="prompts", shape=(1,), dtype=bytes),
            Tensor(name="request_output_len", shape=(1,), dtype=np.uint32),
            Tensor(name="beam_width", shape=(1,), dtype=np.uint32, optional=True),
        )
        outputs = (
            Tensor(name="sentences", shape=(-1, 1), dtype=bytes),
            Tensor(name="sequence_length", shape=(-1,), dtype=np.uint32),
            Tensor(name="cum_log_probs", shape=(-1,), dtype=np.float32),
        )
        self._inputs = {spec.name: spec for spec in inputs}
        self._outputs = {spec.name: spec for spec in outputs}
        self._config = ModelConfig(max_batch_size=128)
        self._tokenizer = tokenizer

    def serve(self, ip: str, http_port=None, grpc_port=None, metrics_port=None):

        triton_config = TritonConfig(
            http_address=ip,
            http_port=http_port,
            grpc_address=ip,
            grpc_port=grpc_port,
            metrics_port=metrics_port,
        )
        with Triton(config=triton_config) as triton:
            triton.bind(
                model_name=self._model_name,
                infer_func=self._infer_fn,
                inputs=tuple(self._inputs.values()),
                outputs=tuple(self._outputs.values()),
                config=self._config,
            )
            triton.serve()

    @batch
    def _infer_fn(self, **inputs: np.ndarray) -> Dict[str, np.ndarray]:
        prompts = np.char.decode(inputs.pop("prompts").astype("bytes"), encoding="utf-8")
        prompts = np.squeeze(prompts, axis=-1).tolist()

        # TODO: group by beam_width and tokens_to_generate
        # beam_width and tokens_to_generate cannot be defined per sample
        batch_size = len(prompts)
        beam_width = inputs.pop("beam_width", np.ones([batch_size, 1]))[0][0]
        output_len = inputs.pop("request_output_len")[0][0]

        prompts_encoding = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._model.weights.max_seq_len - output_len,  # FIXME: not sure if it is the same
        )

        start_lengths = torch.sum(prompts_encoding.attention_mask, axis=-1)
        with torch.no_grad():
            try:
                start_ids = prompts_encoding.input_ids.to(torch.int32)
                start_lengths = start_lengths.to(torch.int32)

                tensors = {"start_ids": start_ids, "start_lengths": start_lengths}
                parameters = {
                    "output_len": output_len,
                    "beam_width": beam_width,
                    "return_output_length": True,
                    "return_cum_log_probs": True,
                }
                broadcast_inputs(tensors, [parameters])
                outputs_ids, _, output_cum_log_probs = self._model(**tensors, **parameters)

            except RuntimeError as e:
                raise PytritonUnrecoverableError("Fatal error occurred - no further inferences possible.") from e

        original_shape = outputs_ids.shape
        outputs_ids = outputs_ids.reshape((-1, outputs_ids.shape[-1]))  # stack batch_axis with beam_axis
        sentences = self._tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)

        sentences = np.array(sentences)
        sentences = sentences.reshape(original_shape[:-1] + (-1,))  # form (batch_axis, beam_axis, 1)
        sentences_length = np.char.str_len(sentences)
        sentences = np.char.encode(sentences, encoding="utf-8").astype(object)

        return {
            "sentences": sentences,
            "sequence_length": sentences_length,
            "cum_log_probs": output_cum_log_probs.cpu().numpy(),
        }

    @classmethod
    def get_remote_kwargs(cls):
        shared_parameters_list = [None]
        broadcast_inputs(cls._shared_tensors_dict, shared_parameters_list)
        (shared_parameters,) = shared_parameters_list
        if shared_parameters:
            return {**cls._shared_tensors_dict, **shared_parameters}
        else:
            return cls._shared_tensors_dict.copy()
