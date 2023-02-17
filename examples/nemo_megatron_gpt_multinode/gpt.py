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
import dataclasses
import typing

import numpy as np
import torch  # pytype: disable=import-error
from nemo.collections.nlp.modules.common.text_generation_utils import generate  # pytype: disable=import-error

from pytriton.decorators import batch, fill_optionals, first_value, group_by_values
from pytriton.exceptions import PyTritonUnrecoverableError
from pytriton.model_config import Tensor

from helpers import cast_output  # pytype: disable=import-error # isort:skip


@dataclasses.dataclass
class ModelParameters:
    tokens_to_generate: np.ndarray = np.array([64], dtype=np.int32)
    min_tokens_to_generate: np.ndarray = np.array([0], dtype=np.int32)
    all_probs: np.ndarray = np.array([False], dtype=np.bool_)
    temperature: np.ndarray = np.array([1.0], dtype=np.float32)
    add_BOS: np.ndarray = np.array([False], dtype=np.bool_)  # noqa: N815
    top_k: np.ndarray = np.array([0], dtype=np.int32)
    top_p: np.ndarray = np.array([0.9], dtype=np.float32)
    greedy: np.ndarray = np.array([False], dtype=np.bool_)
    repetition_penalty: np.ndarray = np.array([1.2], dtype=np.float32)


_PARAMETERS_NAMES = tuple(field.name for field in dataclasses.fields(ModelParameters))


class NemoGptCallable:
    def __init__(self, *, model_name: str, model):
        self.model_name = model_name
        self._model = model.cuda()
        self.inputs = (
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
        self.outputs = (
            Tensor(name="sentences", shape=(1,), dtype=bytes),
            Tensor(name="tokens", shape=(-1,), dtype=bytes),
            Tensor(name="logprob", shape=(-1,), dtype=np.float32),
            Tensor(name="token_ids", shape=(-1,), dtype=np.int32),
            Tensor(name="offsets", shape=(-1,), dtype=np.int32),
        )
        self._outputs_dict = {output.name: output for output in self.outputs}

    @fill_optionals(**dataclasses.asdict(ModelParameters()))
    @batch
    @group_by_values(*_PARAMETERS_NAMES)
    @first_value(*_PARAMETERS_NAMES)
    def infer(self, **inputs: np.ndarray) -> typing.Dict[str, np.ndarray]:
        # Tell other ranks we're doing generate
        generate_num = 0
        choice = torch.cuda.LongTensor([generate_num])
        torch.distributed.broadcast(choice, 0)

        sentences = inputs.pop("sentences").astype("bytes")
        sentences = np.char.decode(sentences, encoding="utf-8")
        sentences = np.squeeze(sentences, axis=-1).tolist()
        try:
            output = generate(self._model, inputs=sentences, **inputs)
        except RuntimeError as e:
            raise PyTritonUnrecoverableError("Fatal error occurred - no further inferences possible.") from e

        output = {
            output_name: cast_output(data, self._outputs_dict[output_name].dtype)
            for output_name, data in output.items()
            if output_name in self._outputs_dict
        }
        return output
