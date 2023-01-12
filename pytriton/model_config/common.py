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
from typing import Dict, Optional


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
