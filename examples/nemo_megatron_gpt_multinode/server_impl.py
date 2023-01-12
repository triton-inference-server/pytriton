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
import dataclasses
from typing import Dict, Sequence

import numpy as np
import torch  # pytype: disable=import-error
from nemo.collections.nlp.modules.common.text_generation_utils import generate  # pytype: disable=import-error

from pytriton.decorators import batch
from pytriton.exceptions import PytritonUnrecoverableError
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig


def unpack_input_batches_by_parameters(_inputs, parameters_names):
    _parameters = [np.squeeze(_inputs[param_name], axis=-1) for param_name in parameters_names]
    _parameters = list(zip(*_parameters))  # transpose parameters
    _unique_parameters = list(set(_parameters))
    _parameters_indices = np.array(list(map(_unique_parameters.index, _parameters)))

    inputs_batches_per_parameter = [
        {
            input_name: data[_parameters_indices == parameter_idx]
            for input_name, data in _inputs.items()
            if input_name not in parameters_names
        }
        for parameter_idx in range(len(_unique_parameters))
    ]

    return inputs_batches_per_parameter, _unique_parameters, _parameters_indices


def _cast_output(data, required_dtype):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    elif not isinstance(data, np.ndarray):
        data = np.array(data)

        data_is_str = required_dtype in (object, np.object_, bytes, np.bytes_)
        if data_is_str:
            data = np.char.encode(data, "utf-8")

    if data.ndim < 2:
        data = data[..., np.newaxis]
    return data.astype(required_dtype)


def pack_output_batches(_outputs: Sequence[Dict[str, np.ndarray]], _indices: np.ndarray) -> Dict[str, np.ndarray]:
    batch_size = len(_indices)
    target = {}
    for parameter_idx, _output in enumerate(_outputs):
        if not target:
            target = {
                output_name: np.zeros((batch_size,) + _data.shape[1:], dtype=_data.dtype)
                for output_name, _data in _output.items()
            }
        for output_name, _data in _output.items():
            target[output_name][_indices == parameter_idx] = _data
    return target


@dataclasses.dataclass
class ModelParameters:
    tokens_to_generate: int = 64
    min_tokens_to_generate: int = 0
    all_probs: bool = False
    temperature: float = 1.0
    add_BOS: bool = False  # noqa
    top_k: int = 0
    top_p: float = 0.9
    greedy: bool = False
    repetition_penalty: float = 1.2


class MegatronTritonServer:
    def __init__(self, model, model_name="GPT"):
        self._model = model.cuda()
        self._model_name = model_name
        inputs = (
            Tensor(name="sentences", shape=(1,), dtype=bytes),
            Tensor(name="tokens_to_generate", shape=(1,), dtype=np.int32, optional=True),
            Tensor(name="min_tokens_to_generate", shape=(1,), dtype=np.int32, optional=True),
            Tensor(name="all_probs", shape=(1,), dtype=np.bool_, optional=True),
            Tensor(name="temperature", shape=(1,), dtype=np.float32, optional=True),
            Tensor(name="add_BOS", shape=(1,), dtype=np.bool_, optional=True),
            Tensor(name="top_k", shape=(1,), dtype=np.int32, optional=True),
            Tensor(name="top_p", shape=(1,), dtype=np.float32, optional=True),
            Tensor(name="greedy", shape=(1,), dtype=np.bool_, optional=True),
            Tensor(name="repetition_penalty", shape=(1,), dtype=np.float32, optional=True),
        )
        outputs = (
            Tensor(name="sentences", shape=(1,), dtype=bytes),
            Tensor(name="tokens", shape=(-1,), dtype=bytes),
            Tensor(name="logprob", shape=(-1,), dtype=np.float32),
            Tensor(name="token_ids", shape=(-1,), dtype=np.int32),
            Tensor(name="offsets", shape=(-1,), dtype=np.int32),
        )
        self._inputs = {spec.name: spec for spec in inputs}
        self._outputs = {spec.name: spec for spec in outputs}
        self._config = ModelConfig(max_batch_size=128)
        self._parameters_names = tuple(field.name for field in dataclasses.fields(ModelParameters))

    def serve(self, ip: str, http_port=None, grpc_port=None, metrics_port=None):

        triton_config = TritonConfig(
            http_address=ip,
            http_port=http_port,
            grpc_address=ip,
            grpc_port=grpc_port,
            metrics_port=metrics_port,
            log_verbose=4,
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
        # Tell other ranks we're doing generate
        generate_num = 0
        choice = torch.cuda.LongTensor([generate_num])
        torch.distributed.broadcast(choice, 0)

        inputs_batches, parameters_per_batch, indices = unpack_input_batches_by_parameters(
            inputs, self._parameters_names
        )

        outputs = []
        for inputs_batch, parameters in zip(inputs_batches, parameters_per_batch):
            sentences = np.char.decode(inputs_batch["sentences"].astype("bytes"), encoding="utf-8")
            sentences = np.squeeze(sentences, axis=-1).tolist()
            parameters = dict(zip(self._parameters_names, parameters))
            try:
                output = generate(self._model, inputs=sentences, **parameters)
            except RuntimeError as e:
                raise PytritonUnrecoverableError("Fatal error occurred - no further inferences possible.") from e
            output = {
                output_name: _cast_output(data, self._outputs[output_name].dtype)
                for output_name, data in output.items()
                if output_name in self._outputs
            }
            outputs.append(output)

        outputs = pack_output_batches(outputs, indices)
        return outputs
