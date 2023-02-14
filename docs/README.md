<!--
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

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

# NVIDIA PyTriton


The NVIDIA PyTriton is Flask/FastAPI-like interface to simplify Triton's deployment in Python environments.
The library allows to serve Machine Learning models directly from Python through
NVIDIA [Triton Inference Server](https://github.com/triton-inference-server).

## How it works?

In NVIDIA PyTriton, same as in Flask or FastAPI, you can define any Python function that execute a Machine Learning model prediction and expose
it through HTTP/gRPC API. The library installs Triton Inference Server in your environment and use it for handling the
HTTP/gRPC requests and responses. Our library provides a Python API that allow to attach a Python function to the Triton
and a communication layer to send/receive data between Triton and the function. The solution helps use of the
performance features of Triton Inference Server, like dynamic batching or response cache, without changing your model
environment. Thus, improve the performance of running inference on GPU of models implemented in Python. The solution is
framework-agnostic and can be used along with frameworks like PyTorch, TensorFlow or JAX.

## Architecture

The diagram below presents the schema how the Python models are served through Triton Inference Server using the
library. The solutions consist of two main components:

- Triton Inference Server - for exposing the HTTP/gRPC API, and benefits from performance features like dynamic batching
  or response cache
- Python Model Environment - your environment where Python model is implemented

The Triton Inference Server binaries are provided as part of the Python package installation. The Triton Server is
installed in your current environment (system or container). The Python library controls the Triton Server process
through `Triton Controller`.

Exposing the model through Triton requires definition of `Inference Callback` - a Python function that is
connected to Triton Inference Server and execute model predictions. The integration layer bind `Inference Callback` and
expose it through Triton HTTP/gRPC API under a provided `<model name>`. Once the integration is done, the defined
`Inference Callback` receive data send to HTTP/gRPC API endpoint `v2/models/<model name>/infer`. Read more
about HTTP/gRPC interface in Triton Inference
Server [documentation](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/inference_protocols.md#httprest-and-grpc-protocols)
.

The `Proxy Model` and `Callback Handler` are responsible for communication between `Inference Callback`
and Triton Inference Server:

- The `Proxy Model` is a binding created on Triton Server side
- The `Callback Handler` is Python thread responsible for send/receive data to/from Inference Callback

A unique `Proxy Model` and `Callback Handler` is created per `Inference Callback`.

When the integration is done the HTTP/gRPC requests send to `v2/models/<model name>/infer` are handled by Triton
Inference Server. The server batch requests for processing and redirect them to appropriate `Proxy Model`.
The `Proxy Model` receive data in form of `numpy` array and send it to the `Inference Callback` through
the `Callback Handler`. Once the `Inference Callback` finished execution of model prediction the result is returned
through the same route and response is created on by Triton.

## Serving the models

The NVIDIA PyTriton provide an option to serve your Python model using Triton Inference Server to
handle HTTP/gRPC
requests and pass the input/output tensors to and from the model. We it a blocking mode where the application is a
long-lived process deployed in your cluster to serve the requests from clients.

Before you run the model for serving the inference callback function has to be defined. Inference callback receive the
inputs and should return the model outputs:

```python
import numpy as np
from pytriton.decorators import batch


@batch
def infer_fn(**inputs: np.ndarray):
    input1, input2 = inputs.values()
    outputs = model(input1, input2)
    return [outputs]
```

The `infer_fn` receive the batched input data for the model and should return the batched outputs.

In the next step you need to create a connection between Triton and the model. On that purpose the `Triton` class has to
be used and the `bind` method is required to be called to create a dedicated connection between Triton Inference
Server the defined `infer_fn`.

In the blocking mode we suggest to use the `Triton` object as a context manager where multiple model can be loaded in
the way presented below:

<!--pytest-codeblocks:cont-->

```python
from pytriton.triton import Triton
from pytriton.model_config import ModelConfig, Tensor

with Triton() as triton:
    triton.bind(
        model_name="MyModel",
        infer_func=infer_fn,
        inputs=[
            Tensor(dtype=bytes, shape=(1,)),  # sample containing single bytes value
            Tensor(dtype=bytes, shape=(-1,)),  # sample containing vector of bytes
        ],
        outputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=16),
    )
```

At this point you have defined how the model has to be handled by Triton and where the HTTP/gRPC requests for model have
to be directed. The last part for serving the model is call the `serve` method on Triton object:

<!--pytest.mark.skip-->

```python
with Triton() as triton:
    # ...
    triton.serve()
```

When `.sever()`  method is called on `Triton` object, the inference queries can be sent to
`localhost:8000/v2/models/MyModel` and the `infer_fn` is called to handle the inference query.

## Working in the Jupyter Notebook

The package provide an option to work with your model inside the Jupyter Notebook. We call it a
background mode where
the model is deployed on Triton Inference Server for handling HTTP/gRPC request but there are other actions that you
want to perform after loading and starting serving the model.

Having the `infer_fn` defined in the same way as described in [serving the models](#serving-the-models) section you
can use the `Triton` object without a context:

```python
from pytriton.triton import Triton
triton = Triton()
```

In the next step the model has to be loaded for serving in Triton Inference Server (which is also the same
as in serving example):

<!--pytest-codeblocks:cont-->

```python
import numpy as np
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor

@batch
def infer_fn(**inputs: np.ndarray):
    input1, input2 = inputs.values()
    outputs = input1 + input2
    return [outputs]

triton.bind(
    model_name="MyModel",
    infer_func=infer_fn,
    inputs=[
        Tensor(shape=(1,), dtype=np.float32),
        Tensor(shape=(-1,), dtype=np.float32),
    ],
    outputs=[Tensor(shape=(-1,), dtype=np.float32)],
    config=ModelConfig(max_batch_size=16),
)
```

Finally, to run the model in background mode use the `run` method:

<!--pytest.mark.skip-->

```python
triton.run()
```

When `.run()`  method is called on `Triton` object, the inference queries can be sent to
`localhost:8000/v2/models/MyModel` and the `infer_fn` is called to handle the inference query.

The Triton server can be stopped at any time using the `stop` method:

<!--pytest.mark.skip-->

```python
triton.stop()
```

## What next?

Read more about using the NVIDIA PyTriton in [quick start](quick_start.md), [examples](examples.md) and
find more options how to configure Triton, models and deployment on cluster in [deploying models](deploying_models.md)
section.

The details about classes and methods you can find in [API reference](api.md) section.
