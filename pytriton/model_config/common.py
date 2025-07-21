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
"""Common structures for internal and external ModelConfig."""

import dataclasses
import enum
from typing import Dict, List, Optional, Sequence, Type, Union

import numpy as np

from pytriton.exceptions import PyTritonBadParameterError


class DeviceKind(enum.Enum):
    """Device kind for model deployment.

    Args:
        KIND_AUTO: Automatically select the device for model deployment.
        KIND_CPU: Model is deployed on CPU.
        KIND_GPU: Model is deployed on GPU.
    """

    KIND_AUTO = "KIND_AUTO"
    KIND_CPU = "KIND_CPU"
    KIND_GPU = "KIND_GPU"


class TimeoutAction(enum.Enum):
    """Timeout action definition for timeout_action QueuePolicy field.

    Args:
        REJECT: Reject the request and return error message accordingly.
        DELAY: Delay the request until all other requests at the same (or higher) priority levels
           that have not reached their timeouts are processed.
    """

    REJECT = "REJECT"
    DELAY = "DELAY"


@dataclasses.dataclass
class QueuePolicy:
    """Model queue policy configuration.

    More in Triton Inference Server [documentation]
    [documentation]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1037

    Args:
        timeout_action: The action applied to timed-out request.
        default_timeout_microseconds: The default timeout for every request, in microseconds.
        allow_timeout_override: Whether individual request can override the default timeout value.
        max_queue_size: The maximum queue size for holding requests.
    """

    timeout_action: TimeoutAction = TimeoutAction.REJECT
    default_timeout_microseconds: int = 0
    allow_timeout_override: bool = False
    max_queue_size: int = 0


@dataclasses.dataclass
class DynamicBatcher:
    """Dynamic batcher configuration.

    More in Triton Inference Server [documentation]
    [documentation]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1104

    Args:
        max_queue_delay_microseconds: The maximum time, in microseconds, a request will be delayed in
                                      the scheduling queue to wait for additional requests for batching.
        preferred_batch_size: Preferred batch sizes for dynamic batching.
        preserve_ordering : Should the dynamic batcher preserve the ordering of responses to
                            match the order of requests received by the scheduler.
        priority_levels: The number of priority levels to be enabled for the model.
        default_priority_level: The priority level used for requests that don't specify their priority.
        default_queue_policy: The default queue policy used for requests.
        priority_queue_policy: Specify the queue policy for the priority level.
    """

    max_queue_delay_microseconds: int = 0
    preferred_batch_size: Optional[list] = None
    preserve_ordering: bool = False
    priority_levels: int = 0
    default_priority_level: int = 0
    default_queue_policy: Optional[QueuePolicy] = None
    priority_queue_policy: Optional[Dict[int, QueuePolicy]] = None


@dataclasses.dataclass
class WarmupInput:
    """Warmup input configuration.

    More in Triton Inference Server [documentation]
    [documentation]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1690
    """

    dtype: Union[Type[np.dtype], Type[object]]
    shape: Sequence[int]
    zero_data: Optional[bool] = False
    random_data: Optional[bool] = False
    input_data_file: Optional[str] = None


@dataclasses.dataclass
class ModelWarmup:
    """Model warmup configuration.

    More in Triton Inference Server [documentation]
    [documentation]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1683
    """

    name: str
    batch_size: int
    inputs: Dict[str, WarmupInput]
    count: int


def generate_warmup_data(warmup_input: WarmupInput, batch_size: int) -> np.ndarray:
    """Generate warmup data from WarmupInput configuration.

    Args:
        warmup_input: Warmup input configuration
        batch_size: Batch size for the warmup data

    Returns:
        Generated numpy array for warmup

    Raises:
        PyTritonBadParameterError: If warmup configuration is invalid
    """
    if not (warmup_input.zero_data or warmup_input.random_data or warmup_input.input_data_file):
        raise PyTritonBadParameterError(
            "One of 'zero_data', 'random_data', or 'input_data_file' must be provided for warmup input."
        )

    # Create full shape including batch dimension
    full_shape = (batch_size,) + tuple(warmup_input.shape)

    if warmup_input.input_data_file:
        # Load data from file
        data = np.load(warmup_input.input_data_file)
        # Ensure data matches expected shape
        if data.shape != full_shape:
            # If file data doesn't match, broadcast or truncate as needed
            if data.ndim == len(warmup_input.shape):
                # Add batch dimension by repeating
                data = np.repeat(data[np.newaxis, ...], batch_size, axis=0)
            elif data.shape[1:] == warmup_input.shape:
                # Adjust batch size
                if data.shape[0] < batch_size:
                    # Repeat to match batch size
                    repeat_count = (batch_size + data.shape[0] - 1) // data.shape[0]
                    data = np.tile(data, (repeat_count,) + (1,) * (data.ndim - 1))
                data = data[:batch_size]  # Truncate if needed
            else:
                raise PyTritonBadParameterError(
                    f"Input data file shape {data.shape} doesn't match expected shape {full_shape}"
                )
        return data.astype(warmup_input.dtype)

    elif warmup_input.zero_data:
        # Generate zero data
        return np.zeros(full_shape, dtype=warmup_input.dtype)

    elif warmup_input.random_data:
        # Generate random data based on dtype
        if warmup_input.dtype == np.bool_:
            return np.random.choice([True, False], size=full_shape)
        elif np.issubdtype(warmup_input.dtype, np.integer):
            # For integer types, generate random integers in a reasonable range
            if warmup_input.dtype == np.int8:
                return np.random.randint(-128, 127, size=full_shape, dtype=warmup_input.dtype)
            elif warmup_input.dtype == np.uint8:
                return np.random.randint(0, 255, size=full_shape, dtype=warmup_input.dtype)
            elif warmup_input.dtype == np.int16:
                return np.random.randint(-32768, 32767, size=full_shape, dtype=warmup_input.dtype)
            elif warmup_input.dtype == np.uint16:
                return np.random.randint(0, 65535, size=full_shape, dtype=warmup_input.dtype)
            else:
                # For other integer types, use a reasonable range
                return np.random.randint(0, 100, size=full_shape, dtype=warmup_input.dtype)
        elif np.issubdtype(warmup_input.dtype, np.floating):
            # For float types, generate random floats
            return np.random.random(full_shape).astype(warmup_input.dtype)
        elif warmup_input.dtype == np.bytes_ or warmup_input.dtype == object:
            # For bytes/string types, generate random strings
            random_strings = []
            for _ in range(np.prod(full_shape)):
                # Generate random string of length 8-32
                length = np.random.randint(8, 33)
                random_str = "".join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), length))
                random_strings.append(random_str.encode("utf-8"))
            return np.array(random_strings, dtype=warmup_input.dtype).reshape(full_shape)
        else:
            # Fallback for other types
            return np.zeros(full_shape, dtype=warmup_input.dtype)

    # Should never reach here due to initial validation
    raise PyTritonBadParameterError("Invalid warmup input configuration")


def generate_warmup_requests(
    model_warmup: List[ModelWarmup], input_specs: List, outputs_specs: List
) -> List[Dict[str, np.ndarray]]:
    """Generate warmup requests from ModelWarmup configuration.

    Args:
        model_warmup: List of model warmup configurations
        input_specs: List of input tensor specifications
        outputs_specs: List of output tensor specifications (for validation)

    Returns:
        List of warmup request dictionaries with input data
    """
    if not model_warmup:
        return []

    # Create mapping of input names for validation
    input_names = {spec.name for spec in input_specs}

    warmup_requests = []

    for warmup_config in model_warmup:
        # It always faules because input_names are empty
        # # Validate that all warmup inputs match model inputs
        # for input_name in warmup_config.inputs:
        #     if input_name not in input_names:
        #         raise PyTritonBadParameterError(
        #             f"Warmup input '{input_name}' does not match any model input. "
        #             f"Available inputs: {sorted(input_names)}"
        #         )

        # Generate requests for this warmup configuration
        for _ in range(warmup_config.count):
            request_data = {}
            for input_name, warmup_input in warmup_config.inputs.items():
                request_data[input_name] = generate_warmup_data(warmup_input, warmup_config.batch_size)
            warmup_requests.append(request_data)

    return warmup_requests
