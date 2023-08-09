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
"""Model configurations.

Dataclasses with specialized deployment paths for models on Triton. The purpose of this module is to provide clear options
to configure models of given types.

The dataclasses are exposed in the user API.
"""
import dataclasses

from pytriton.model_config import DynamicBatcher


@dataclasses.dataclass
class ModelConfig:
    """Additional model configuration for running model through Triton Inference Server.

    Args:
        batching: Flag to enable/disable batching for model.
        max_batch_size: The maximal batch size that would be handled by model.
        batcher: Configuration of Dynamic Batching for the model.
        response_cache: Flag to enable/disable response cache for the model
        decoupled: Flag to enable/disable decoupled from requests execution
    """

    batching: bool = True
    max_batch_size: int = 4
    batcher: DynamicBatcher = dataclasses.field(default_factory=DynamicBatcher)
    response_cache: bool = False
    decoupled: bool = False
