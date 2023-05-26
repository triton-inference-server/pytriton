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

This document provides guidelines for creating an inference callable for PyTriton, which serves as the entry point for
handling inference requests.

The inference callable is an entry point for handling inference requests. The interface of the inference callable
assumes it receives a list of requests with input dictionaries, where each dictionary represents one request mapping model input
names to NumPy ndarrays.
Requests contain also custom HTTP/gRPC headers and parameters in parameters dictionary.

## Function

The simples inference callable is a function that implement the interface to handle request and responses.
Request class contains following fields:
- data - for inputs (stored as dictionary, but can be also accessed with request dict interface e.g. request["input_name"])
- parameters - for combined parameters and HTTP/gRPC headers
For more information about parameters and headers see [here](custom_params.md).

 ```python
 import numpy as np
 from typing import Dict, List
 from pytriton.proxy.types import Request

 def infer_fn(requests: List[Request]) -> List[Dict[str, np.ndarray]]:
     ...
 ```

## Class

In many cases is worth to use an object of given class as callable. This is especially useful when you want to have a
control over the order of initialized objects or models.

 <!--pytest-codeblocks:cont-->

 ```python
 import numpy as np
 from typing import Dict, List
 from pytriton.proxy.types import Request

 class InferCallable:

     def __call__(self, requests: List[Request]) -> List[Dict[str, np.ndarray]]:
        ...
 ```

## Binding to Triton

To use the inference callable with PyTriton, it must be bound to a Triton server instance using the `bind` method:

<!--pytest-codeblocks:cont-->

```python
import numpy as np
from pytriton.triton import Triton
from pytriton.model_config import ModelConfig, Tensor

with Triton() as triton:
    triton.bind(
        model_name="MyInferenceFn",
        infer_func=infer_fn,
        inputs=[Tensor(shape=(1,), dtype=np.float32)],
        outputs=[Tensor(shape=(1,), dtype=np.float32)],
        config=ModelConfig(max_batch_size=8)
    )

    infer_callable = InferCallable()
    triton.bind(
        model_name="MyInferenceCallable",
        infer_func=infer_callable,
        inputs=[Tensor(shape=(1,), dtype=np.float32)],
        outputs=[Tensor(shape=(1,), dtype=np.float32)],
        config=ModelConfig(max_batch_size=8)
    )
```

For more information on serving the inference callable, refer to
the [Loading models section](binding_models.md) on Deploying Models page.
