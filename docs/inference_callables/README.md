<!--
Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.

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

# Inference Callable

This document provides guidelines for creating an Inference Callable for PyTriton, which serves as the entry point for handling inference requests.

## Overview

The simplest Inference Callable is a function that implements the interface to handle requests and returns responses.

The [Request][pytriton.proxy.types.Request] class contains the following fields:

- `data` - for inputs stored as a mapping. It can also be accessed with the request mapping protocol of the `Request` object (e.g., request["input_name"])
- `parameters` - for mapping consisting of combined parameters and HTTP/gRPC headers

For more information about parameters and headers, see [here](custom_params.md).

```python
import numpy as np
from typing import Dict, List
from pytriton.proxy.types import Request

def infer_fn(requests: List[Request]) -> List[Dict[str, np.ndarray]]:
    ...
```

In many cases, it is worth implement Inference Callable as method. This is especially useful when you want to have control over pipeline instance initialization.

<!--pytest-codeblocks:cont-->

```python
import numpy as np
from typing import Dict, List
from pytriton.proxy.types import Request

class InferCallable:

    def __init__(self, *args, **kwargs):
        ...  # model initialization

    def __call__(self, requests: List[Request]) -> List[Dict[str, np.ndarray]]:
        ...

    def alternative_infer(self, requests: List[Request]) -> List[Dict[str, np.ndarray]]:
        ...
```

### Asynchronous Interface

Some models can run asynchronously, meaning they can process multiple requests at the same time without waiting for each one to finish. If your model supports this feature, you can use an asynchronous coroutine to define the Inference Callable.

<!--pytest-codeblocks:cont-->

```python
import numpy as np
from typing import Dict, List
from pytriton.proxy.types import Request

async def infer_coro(requests: List[Request]) -> List[Dict[str, np.ndarray]]:
    ...
```

### Streaming Partial Results

Some models can send more than one response for a request, or no response at all. This is useful for models that produce intermediate results or stream data continuously. If your model supports this feature, you can use a generator function or an asynchronous generator coroutine to define the Inference Callable:

<!--pytest-codeblocks:cont-->

```python
import numpy as np
from typing import Dict, Generator, List
from pytriton.proxy.types import Request

def streaming_infer_fn(requests: List[Request]) -> Generator[Dict[str, np.ndarray], None, None]:
    ...
```

This feature only works when the model is served in decoupled mode. For more information, see the [Decoupled Mode](decoupled_mode.md) section.

## Binding to Triton

To use the Inference Callable with PyTriton, it must be bound to a Triton server instance using the `bind` method.

 <!--pytest-codeblocks:cont-->

```python
import numpy as np
from pytriton.triton import Triton
from pytriton.model_config import ModelConfig, Tensor

infer_callable = InferCallable()

with Triton() as triton:
    triton.bind(
        model_name="MyInferenceFunction",
        infer_func=infer_fn,
        inputs=[Tensor(shape=(1,), dtype=np.float32)],
        outputs=[Tensor(shape=(1,), dtype=np.float32)],
        config=ModelConfig(max_batch_size=8)
    )
    triton.bind(
        model_name="MyInferenceCallable",
        infer_func=infer_callable,
        inputs=[Tensor(shape=(1,), dtype=np.float32)],
        outputs=[Tensor(shape=(1,), dtype=np.float32)],
        config=ModelConfig(max_batch_size=8)
    )
    triton.bind(
        model_name="AlternativeInferenceCallable",
        infer_func=infer_callable.alternative_infer,
        inputs=[Tensor(shape=(1,), dtype=np.float32)],
        outputs=[Tensor(shape=(1,), dtype=np.float32)],
        config=ModelConfig(max_batch_size=8)
    )
    triton.bind(
        model_name="MyInferenceCoroutine",
        infer_func=infer_coro,
        inputs=[Tensor(shape=(1,), dtype=np.float32)],
        outputs=[Tensor(shape=(1,), dtype=np.float32)],
        config=ModelConfig(max_batch_size=8)
    )
    triton.bind(
        model_name="StreamingInferenceCallable",
        infer_func=streaming_infer_fn,
        inputs=[Tensor(shape=(1,), dtype=np.float32)],
        outputs=[Tensor(shape=(1,), dtype=np.float32)],
        config=ModelConfig(max_batch_size=8, decoupled=True)
    )
```

For more information on serving the Inference Callable, refer to the [Loading models section](binding_models.md) on the Deploying Models page.
