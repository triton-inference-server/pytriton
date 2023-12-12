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
"""Validators used in proxy module."""
import logging

import numpy as np

from pytriton.proxy.types import Requests, Responses

LOGGER = logging.getLogger(__name__)


class TritonResultsValidator:
    """Validate results returned by inference callable against PyTriton and Triton requirements."""

    def __init__(self, model_config, strict: bool):
        """Validate results returned by inference callable against PyTriton and Triton requirements.

        Args:
            model_config: Model configuration on Triton side
            strict: Enable/disable strict validation against model config
        """
        self._model_config = model_config
        self._model_outputs = {output.name: output for output in model_config.outputs}
        self._strict = strict

    def validate_responses(self, requests: Requests, responses: Responses):
        """Validate responses returned by inference callable against PyTriton and Triton requirements.

        Args:
            requests: Requests received from Triton
            responses: Responses returned by inference callable

        Raises:
            ValueError if responses are incorrect
        """
        requests_number = len(requests)
        _validate_outputs(self._model_config, self._model_outputs, responses, self._strict, requests_number)


def _validate_outputs(model_config, model_outputs, outputs, strict: bool, requests_number: int):
    """Validate outputs of model.

    Args:
        model_config: Model configuration on Triton side
        model_outputs: Mapped outputs configuration
        outputs: Returned outputs from inference callable
        strict: Enable/disable strict validation against model config
        requests_number: Number of requests

    Raises:
        ValueError if outputs are incorrect
    """
    if not isinstance(outputs, list):
        raise ValueError(
            f"Outputs returned by `{model_config.model_name}` model callable "
            f"must be list of response dicts with numpy arrays. Got outputs={outputs} instead."
        )
    if len(outputs) != requests_number:
        raise ValueError(
            f"Number of outputs returned by `{model_config.model_name}` inference callable "
            f"({len(outputs)}) does not match number of requests ({requests_number}) received from Triton."
        )

    LOGGER.debug(f"Number of responses: {len(outputs)}")
    for response_idx, response in enumerate(outputs):
        LOGGER.debug(f"Response #{response_idx}")
        if not isinstance(response, dict):
            raise ValueError(
                f"Outputs returned by `{model_config.model_name}` model callable "
                f"must be list of response dicts with numpy arrays. Got response={response} instead."
            )
        for name, value in response.items():
            LOGGER.debug(f"    {name}: {value} shape={value.shape} dtype={value.dtype}")
            _validate_output_data(model_config, name, value)
            if strict:
                _validate_output_dtype_and_shape(model_config, model_outputs, name, value)


def _validate_output_data(model_config, name, value):
    """Validate output with given name and value.

    Args:
        model_config: Model configuration on Triton side
        name: Name of output
        value: Value returned in output

    Raises:
        ValueError if output is incorrect
    """
    if not isinstance(name, str):
        raise ValueError(f"Not all keys returned by `{model_config.model_name}` model callable are string")
    if not isinstance(value, np.ndarray):
        raise ValueError(f"Not all values returned by `{model_config.model_name}` model callable are numpy arrays")
    else:
        allowed_kind = "biufOSU"
        if value.dtype.kind not in allowed_kind:
            raise ValueError(
                f"Only bool, numeric, string, unicode and object arrays "
                f"are supported by Triton (dtype.kind: {allowed_kind}). "
                f"Returned `{name}` for model `{model_config.model_name}` "
                f"has `{value.dtype.kind}` dtype.kind."
            )
        if value.dtype.kind == "O":
            if isinstance(value.item(0), str):
                raise ValueError(
                    "Use string/byte-string instead of object for passing "
                    f"string in NumPy array from model `{model_config.model_name}`."
                )
            elif not isinstance(value.item(0), bytes):
                raise ValueError(
                    "Only bytes as objects dtype are supported by PyTriton. "
                    f"Returned `{name}` from `{model_config.model_name}` "
                    f"has `{type(value.item(0))}` type."
                )


def _validate_output_dtype_and_shape(model_config, model_outputs, name, value):
    """Validate output with given name and value against the model config.

    Args:
        model_config: Model configuration on Triton side
        model_outputs: Mapped outputs defined in model config
        name: Name of output
        value: Value returned in output

    Raises:
        ValueError if output does not match defined values in model config
    """
    output_config = model_outputs.get(name)
    if not output_config:
        raise ValueError(
            f"Returned output `{name}` is not defined in model config for model `{model_config.model_name}`."
        )

    allowed_object_types = [bytes, object, np.bytes_, np.object_]
    if (value.dtype.kind not in "OSU" and not np.issubdtype(value.dtype, output_config.dtype)) or (
        value.dtype.kind in "OSU" and output_config.dtype not in allowed_object_types
    ):
        raise ValueError(
            f"Returned output `{name}` for model `{model_config.model_name}` has invalid type. "
            f"Returned: {value.dtype} ({value.dtype.kind}). Expected: {output_config.dtype}."
        )

    batch_shape = 1 if model_config.batching else 0
    if len(value.shape[batch_shape:]) != len(output_config.shape):
        raise ValueError(
            f"Returned output `{name}` for model `{model_config.model_name}` has invalid shapes. "
            f"Returned: {value.shape[batch_shape:]}. Expected: {output_config.shape}."
        )
    if any(x != y != -1 for x, y in zip(value.shape[batch_shape:], output_config.shape)):
        raise ValueError(
            f"Returned output `{name}` for model `{model_config.model_name}` "
            "has invalid shapes at one or more positions. "
            f"Returned: {value.shape[batch_shape:]}. Expected: {output_config.shape}."
        )
