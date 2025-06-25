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
import numpy as np

from pytriton.model_config.common import (
    DeviceKind,
    DynamicBatcher,
    ModelWarmup,
    QueuePolicy,
    TimeoutAction,
    WarmupInput,
)
from pytriton.model_config.triton_model_config import ResponseCache, TensorSpec, TritonModelConfig

full_model_config = TritonModelConfig(
    model_name="simple",
    batching=True,
    max_batch_size=16,
    batcher=DynamicBatcher(
        preferred_batch_size=[16, 32],
        max_queue_delay_microseconds=100,
        preserve_ordering=True,
        priority_levels=3,
        default_priority_level=1,
        default_queue_policy=QueuePolicy(
            allow_timeout_override=True,
            timeout_action=TimeoutAction.DELAY,
            default_timeout_microseconds=100,
            max_queue_size=2,
        ),
        priority_queue_policy={
            2: QueuePolicy(
                allow_timeout_override=True,
                timeout_action=TimeoutAction.DELAY,
                default_timeout_microseconds=100,
                max_queue_size=3,
            )
        },
    ),
    instance_group={DeviceKind.KIND_CPU: 1, DeviceKind.KIND_GPU: 2},
    decoupled=True,
    backend_parameters={
        "parameter1": "value1",
        "parameter2": "value2",
    },
    inputs=[
        TensorSpec(name="INPUT_1", dtype=np.float32, shape=(-1,)),
        TensorSpec(name="INPUT_2", dtype=np.bytes_, shape=(-1,)),
    ],
    outputs=[
        TensorSpec(name="OUTPUT_1", dtype=np.int32, shape=(1000,)),
    ],
    response_cache=ResponseCache(enable=True),
    model_warmup=[
        ModelWarmup(
            name="simple",
            batch_size=16,
            inputs={
                "INPUT_1": WarmupInput(dtype=np.float32, shape=(1,), random_data=True),
                "INPUT_2": WarmupInput(dtype=np.bytes_, shape=(1,), random_data=True),
            },
            count=2,
        ),
    ],
)
