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

LOGGER = logging.getLogger(__name__)


def validate_outputs(model_config, model_outputs, outputs, strict: bool, requests_number: int):
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
            "must be list of response dicts with numpy arrays"
        )
    if len(outputs) != requests_number:
        raise ValueError(
            f"Number of outputs returned by `{model_config.model_name}` inference callable "
            f"({len(outputs)}) does not match number of requests ({requests_number}) received from Triton."
        )

    LOGGER.debug(f"Outputs: {outputs}")
    for response in outputs:
        LOGGER.debug(f"Response: {response}")
        if not isinstance(response, dict):
            raise ValueError(
                f"Outputs returned by `{model_config.model_name}` model callable "
                "must be list of response dicts with numpy arrays"
            )
        for name, value in response.items():
            LOGGER.debug(f"{name}: {value}")
            validate_output_data(model_config, name, value)
            if strict:
                validate_output_dtype_and_shape(model_config, model_outputs, name, value)


def validate_output_data(model_config, name, value):
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


def validate_output_dtype_and_shape(model_config, model_outputs, name, value):
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
    LOGGER.debug(
        f"Current output `{name}` for model `{model_config.model_name}` has shape: {value.shape[batch_shape:]}"
    )
    LOGGER.debug(f"Expected output `{name}` for model `{model_config.model_name}` has shape: {output_config.shape}")
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
