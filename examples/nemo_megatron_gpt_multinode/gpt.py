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
import typing

import numpy as np
import torch  # pytype: disable=import-error
from nemo.collections.nlp.modules.common.transformer.text_generation import (  # pytype: disable=import-error
    LengthParam,
    OutputType,
    SamplingParam,
)

from pytriton.decorators import ConstantPadder, batch, first_value, group_by_values
from pytriton.exceptions import PyTritonInvalidOperationError, PyTritonUnrecoverableError
from pytriton.model_config import Tensor

from helpers import cast_output, typedict2tensor  # pytype: disable=import-error # isort:skip


_INPUT_PARAMETERS_NAMES = list(typing.get_type_hints(LengthParam)) + list(typing.get_type_hints(SamplingParam))
_INPUT_PARAMETERS_NAMES_WITHOUT_END_STRINGS = _INPUT_PARAMETERS_NAMES.copy()
_INPUT_PARAMETERS_NAMES_WITHOUT_END_STRINGS.remove("end_strings")


class NemoGptCallable:
    def __init__(self, *, model_name: str, model):
        self.model_name = model_name
        self._model = model.cuda()
        self._is_prompt_learning_model = hasattr(model, "virtual_prompt_style")
        self._text_generate_fn = (
            self._model.frozen_model.generate if self._is_prompt_learning_model else self._model.generate
        )
        self._task_generate_fn = self._model.generate if self._is_prompt_learning_model else None
        self.inputs = (
            (
                Tensor(name="tasks", shape=(1,), dtype=bytes),
                Tensor(name="prompts", shape=(1,), dtype=bytes),
            )
            + typedict2tensor(LengthParam, overwrite_kwargs={"optional": True}, defaults=None)
            + typedict2tensor(SamplingParam, overwrite_kwargs={"optional": True}, defaults=None)
        )
        self.outputs = typedict2tensor(OutputType)
        self._outputs_dict = {output.name: output for output in self.outputs}

    def _format_prompts(
        self, tasks: typing.List[str], prompts: typing.List[str]
    ) -> typing.List[typing.Union[str, typing.Dict[str, str]]]:
        formatted_prompts = []
        for task_name, prompt in zip(tasks, prompts):
            task_template = self._model.task_templates[task_name]
            formatted_prompts.append(
                {
                    **{"taskname": task_name},
                    **dict(zip(task_template["prompt_template_fields"], [prompt])),
                }
            )
        return formatted_prompts

    @batch
    @group_by_values("tasks", *_INPUT_PARAMETERS_NAMES, pad_fn=ConstantPadder(0))
    @first_value(*_INPUT_PARAMETERS_NAMES_WITHOUT_END_STRINGS)
    def infer(self, **inputs: np.ndarray) -> typing.Dict[str, np.ndarray]:
        # Tell other ranks we're doing generate
        generate_num = 0
        choice = torch.cuda.LongTensor([generate_num])
        torch.distributed.broadcast(choice, 0)

        def _str_ndarray2list(str_ndarray: np.ndarray) -> typing.List[str]:
            str_ndarray = str_ndarray.astype("bytes")
            str_ndarray = np.char.decode(str_ndarray, encoding="utf-8")
            str_ndarray = str_ndarray.squeeze(axis=-1)
            return str_ndarray.tolist()

        tasks = _str_ndarray2list(inputs.pop("tasks"))
        prompts = _str_ndarray2list(inputs.pop("prompts"))
        length_params = LengthParam(**{k: v for k, v in inputs.items() if k in typing.get_type_hints(LengthParam)})
        end_strings = inputs.pop("end_strings")
        #end_strings = [end_strings.decode(encoding="utf-8")]
        end_strings = [es.decode("utf-8") for es in end_strings[0][0]]
        sampling_params = SamplingParam(
            **{k: v for k, v in inputs.items() if k in typing.get_type_hints(SamplingParam)}
        )
        sampling_params["end_strings"] = end_strings
        if tasks[0] == "text_generation":
            generate_fn = self._text_generate_fn
        else:
            generate_fn = self._task_generate_fn
            if generate_fn is None:
                raise PyTritonInvalidOperationError(
                    f"Model {self.model_name} does not support task {inputs['task']}. "
                    "Only text_generation task is supported."
                )
            prompts = self._format_prompts(tasks, prompts)

        try:
            output: OutputType = generate_fn(
                inputs=prompts,
                length_params=length_params,
                sampling_params=sampling_params,
            )
        except RuntimeError as e:
            raise PyTritonUnrecoverableError("Fatal error occurred - no further inferences possible.") from e

        output = {
            output_name: cast_output(data, self._outputs_dict[output_name].dtype)
            for output_name, data in output.items()
        }
        return output
