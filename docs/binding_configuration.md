<!--
Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# Binding Configuration

The additional configuration of binding the model for running a model through the Triton Inference Server can be
provided in the `config` argument in the `bind` method. This section describes the possible configuration enhancements.
The configuration of the model can be adjusted by overriding the defaults for the `ModelConfig` object.

```python
from pytriton.model_config.common import DynamicBatcher

class ModelConfig:
    batching: bool = True
    max_batch_size: int = 4
    batcher: DynamicBatcher = DynamicBatcher()
    response_cache: bool = False
```

## Batching

The batching feature collects one or more samples and passes them to the model together. The model processes
multiple samples at the same time and returns the output for all the samples processed together.

Batching can significantly improve throughput. Processing multiple samples at the same time leverages the benefits of
utilizing GPU performance for inference.

The Triton Inference Server is responsible for collecting multiple incoming requests into a single batch. The batch is
passed to the model, which improves the inference performance (throughput and latency). This feature is called
`dynamic batching`, which collects samples from multiple clients into a single batch processed by the model.

On the PyTriton side, the `infer_fn` obtain the fully created batch by Triton Inference Server so the only
responsibility is to perform computation and return the output.

By default, batching is enabled for the model. The default behavior for Triton is to have dynamic batching enabled.
If your model does not support batching, use `batching=False` to disable it in Triton.

## Maximal batch size

The maximal batch size defines the number of samples that can be processed at the same time by the model. This configuration
has an impact not only on throughput but also on memory usage, as a bigger batch means more data loaded to the memory
at the same time.

The `max_batch_size` has to be a value greater than or equal to 1.

## Dynamic batching

The dynamic batching is a Triton Inference Server feature and can be configured by defining the `DynamicBatcher`
object:

```python
from typing import Dict, Optional
from pytriton.model_config.common import QueuePolicy

class DynamicBatcher:
    max_queue_delay_microseconds: int = 0
    preferred_batch_size: Optional[list] = None
    preserve_ordering: bool = False
    priority_levels: int = 0
    default_priority_level: int = 0
    default_queue_policy: Optional[QueuePolicy] = None
    priority_queue_policy: Optional[Dict[int, QueuePolicy]] = None
```

More about dynamic batching can be found in
the [Triton Inference Server documentation](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher)
and [API spec](api.md)

## Response cache

The Triton Inference Server provides functionality to use a cached response for the model. To use the response cache:

- provide the `cache_config` in `TritonConfig`
- set `response_cache=True` in `ModelConfig`

More about response cache can be found in the [Triton Response Cache](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/response_cache.md) page.

Example:

<!--pytest.mark.skip-->
```python
import numpy as np

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

triton_config = TritonConfig(
    cache_config=[f"local,size={1024 * 1024}"],  # 1MB
)

@batch
def _add_sub(**inputs):
    a_batch, b_batch = inputs.values()
    add_batch = a_batch + b_batch
    sub_batch = a_batch - b_batch
    return {"add": add_batch, "sub": sub_batch}

with Triton(config=triton_config) as triton:
    triton.bind(
        model_name="AddSub",
        infer_func=_add_sub,
        inputs=[Tensor(shape=(1,), dtype=np.float32), Tensor(shape=(1,), dtype=np.float32)],
        outputs=[Tensor(shape=(1,), dtype=np.float32), Tensor(shape=(1,), dtype=np.float32)],
        config=ModelConfig(max_batch_size=8, response_cache=True)
    )
    ...
```
