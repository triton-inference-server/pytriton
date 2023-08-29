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
# Binding Models to Triton

The Triton class provides methods to bind one or multiple models to the Triton server in order to expose HTTP/gRPC
endpoints for inference serving:

```python
import numpy as np
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton


@batch
def infer_fn(**inputs: np.ndarray):
    input1, input2 = inputs.values()
    outputs = model(input1, input2)
    return [outputs]

with Triton() as triton:
  triton.bind(
      model_name="ModelName",
      infer_func=infer_fn,
      inputs=[
          Tensor(shape=(1,), dtype=np.bytes_),  # sample containing single bytes value
          Tensor(shape=(-1,), dtype=np.bytes_)  # sample containing vector of bytes
      ],
      outputs=[
          Tensor(shape=(-1,), dtype=np.float32),
      ],
      config=ModelConfig(max_batch_size=8),
      strict=True,
  )
```

The `bind` method's mandatory arguments are:

- `model_name`: defines under which name the model is available in Triton Inference Server
- `infer_func`: function or Python `Callable` object which obtains the data passed in the request and returns the output
- `inputs`: defines the number, types, and shapes for model inputs
- `outputs`: defines the number, types, and shapes for model outputs
- `config`: more customization for model deployment and behavior on the Triton server
- `strict`: enable inference callable output validation of data types and shapes against provided model config (default: True)

Once the `bind` method is called, the model is created in the Triton Inference Server model store under
the provided `model_name`.

## Inference Callable

The inference callable is an entry point for inference. This can be any callable that receives the data for
model inputs in the form of a list of request dictionaries where input names are mapped into ndarrays.
Input can be also adapted to different more convenient forms using a set of decorators.
**More details about designing inference callable and using of decorators can be found
in [Inference Callable](inference_callable.md) page.**

In the simplest implementation for functionality that passes input data on output, a lambda can be used:

```python
import numpy as np
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

with Triton() as triton:
  triton.bind(
      model_name="Identity",
      infer_func=lambda requests: requests,
      inputs=[Tensor(dtype=np.float32, shape=(1,))],
      outputs=[Tensor(dtype=np.float32, shape=(1,))],
      config=ModelConfig(max_batch_size=8)
  )
```

## Multi-instance model inference

Multi-instance model inference is a mechanism for loading multiple instances of the same model and calling
them alternately (to hide transfer overhead).

With the `Triton` class, it can be realized by providing the list of multiple inference callables to `Triton.bind`
in the `infer_func` parameter.

The example presents multiple instances of the Linear PyTorch model loaded on separate devices.

First, define the wrapper class for the inference handler. The class initialization receives a model and device as
arguments. The inference handling is done by method `__call__` where the `model` instance is called:

```python
import torch
from pytriton.decorators import batch


class _InferFuncWrapper:
    def __init__(self, model: torch.nn.Module, device: str):
        self._model = model
        self._device = device

    @batch
    def __call__(self, **inputs):
        (input1_batch,) = inputs.values()
        input1_batch_tensor = torch.from_numpy(input1_batch).to(self._device)
        output1_batch_tensor = self._model(input1_batch_tensor)
        output1_batch = output1_batch_tensor.cpu().detach().numpy()
        return [output1_batch]
```

Next, create a factory function where a model and instances of `_InferFuncWrapper` are created - one per each device:

<!--pytest-codeblocks:cont-->

```python
def _infer_function_factory(devices):
    infer_fns = []
    for device in devices:
        model = torch.nn.Linear(20, 30).to(device).eval()
        infer_fns.append(_InferFuncWrapper(model=model, device=device))

    return infer_fns
```

Finally, the list of callable objects is passed to `infer_func` parameter of the `Triton.bind` function:

<!--pytest-codeblocks:cont-->

```python
import numpy as np
from pytriton.triton import Triton
from pytriton.model_config import ModelConfig, Tensor

with Triton() as triton:
  triton.bind(
      model_name="Linear",
      infer_func=_infer_function_factory(devices=["cuda", "cpu"]),
      inputs=[
          Tensor(dtype=np.float32, shape=(-1,)),
      ],
      outputs=[
          Tensor(dtype=np.float32, shape=(-1,)),
      ],
      config=ModelConfig(max_batch_size=16),
  )
  ...
```

Once the multiple callable objects are passed to `infer_func`, the Triton server gets information that multiple instances
of the same model have been created. The incoming requests are distributed among created instances. In our case executing
two instances of a `Linear` model loaded on CPU and GPU devices.

## Defining Inputs and Outputs

The integration of the Python model requires the inputs and outputs types of the model. This is required to
correctly map the input and output data passed through the Triton Inference Server.

The simplest definition of model inputs and outputs expects providing the type of data and the shape per input:

```python
import numpy as np
from pytriton.model_config import Tensor

inputs = [
    Tensor(dtype=np.float32, shape=(-1,)),
]
output = [
    Tensor(dtype=np.float32, shape=(-1,)),
    Tensor(dtype=np.int32, shape=(-1,)),
]
```


The provided configuration creates the following tensors:

- Single input:
  - name: INPUT_1, data type: FLOAT32, shape=(-1,)
- Two outputs:
  - name: OUTPUT_1, data type: FLOAT32, shape=(-1,)
  - name: OUTPUT_2, data type: INT32, shape=(-1,)

The `-1` means a dynamic shape of the input or output.

To define the name of the input and its exact shape, the following definition can be used:

```python
import numpy as np
from pytriton.model_config import Tensor

inputs = [
    Tensor(name="image", dtype=np.float32, shape=(224, 224, 3)),
]
outputs = [
    Tensor(name="class", dtype=np.int32, shape=(1000,)),
]
```

This definition describes that the model has:

- a single input named `image` of size 224x224x3 and 32-bit floating-point data type
- a single output named `class` of size 1000 and 32-bit integer data type.

The `dtype` parameter can be either `numpy.dtype`, `numpy.dtype.type`, or `str`. For example:

```python
import numpy as np
from pytriton.model_config import Tensor

tensor1 = Tensor(name="tensor1", shape=(-1,), dtype=np.float32),
tensor2 = Tensor(name="tensor2", shape=(-1,), dtype=np.float32().dtype),
tensor3 = Tensor(name="tensor3", shape=(-1,), dtype="float32"),
```

!!! warning "dtype for bytes and string inputs/outputs"

    When using the `bytes` dtype, NumPy removes trailing `\x00` bytes.
    Therefore, for arbitrary bytes, it is required to use `object` dtype.

        > np.array([b"\xff\x00"])
        array([b'\xff'], dtype='|S2')

        > np.array([b"\xff\x00"], dtype=object)
        array([b'\xff\x00'], dtype=object)

    For ease of use, for encoded string values, users might use `bytes` dtype.

## Throwing Unrecoverable errors

When the model gets into a state where further inference is impossible,
you can throw [PyTritonUnrecoverableError][pytriton.exceptions.PyTritonUnrecoverableError]
from the inference callable. This will cause NVIDIA Triton Inference Server to shut down.
This might be useful when the model is deployed on a cluster in a multi-node setup. In that case
to recover the model you need to restart all "workers" on the cluster.

When the model gets into a state where further inference is impossible,
you can throw the [PyTritonUnrecoverableError][pytriton.exceptions.PyTritonUnrecoverableError]
from the inference callable. This will cause the NVIDIA Triton Inference Server to shut down.
This might be useful when the model is deployed on a cluster in a multi-node setup. In that case,
to recover the model, you need to restart all "workers" on the cluster.

```python
from typing import Dict
import numpy as np
from pytriton.decorators import batch
from pytriton.exceptions import PyTritonUnrecoverableError


@batch
def infer_fn(**inputs: np.ndarray) -> Dict[str, np.ndarray]:
    ...

    try:
        outputs = model(**inputs)
    except Exception as e:
        raise PyTritonUnrecoverableError(
            "Some unrecoverable error occurred, "
            "thus no further inferences are possible."
        ) from e

    ...
    return outputs
```