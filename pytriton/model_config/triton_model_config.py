# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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
"""ModelConfig related objects."""

import dataclasses
from typing import Dict, Optional, Sequence, Type, Union

import numpy as np

from .common import DeviceKind, DynamicBatcher


@dataclasses.dataclass
class ResponseCache:
    """Model response cache configuration.

    More in Triton Inference Server [documentation]
    [documentation]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto#L1765
    """

    enable: bool


@dataclasses.dataclass
class TensorSpec:
    """Stores specification of single tensor. This includes name, shape and dtype."""

    name: str
    shape: tuple
    dtype: Union[Type[np.dtype], Type[object]]
    optional: Optional[bool] = False


@dataclasses.dataclass
class TritonModelConfig:
    """Triton Model Config dataclass for simplification and specialization of protobuf config generation.

    More in Triton Inference Server [documentation]
    [documentation]: https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto
    """

    model_name: str
    model_version: int = 1
    max_batch_size: int = 4
    batching: bool = True
    batcher: Optional[DynamicBatcher] = None
    instance_group: Dict[DeviceKind, Optional[int]] = dataclasses.field(default_factory=lambda: {})
    decoupled: bool = False
    backend_parameters: Dict[str, str] = dataclasses.field(default_factory=lambda: {})
    inputs: Optional[Sequence[TensorSpec]] = None
    outputs: Optional[Sequence[TensorSpec]] = None
    response_cache: Optional[ResponseCache] = None

    @property
    def backend(self) -> str:
        """Return backend parameter."""
        return "python"
