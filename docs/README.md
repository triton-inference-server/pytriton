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

# PyTriton

PyTriton is a Flask/FastAPI-like interface that simplifies Triton's deployment in Python environments.
The library allows serving Machine Learning models directly from Python through
NVIDIA's [Triton Inference Server](https://github.com/triton-inference-server).

## How it works?

In PyTriton, as in Flask or FastAPI, you can define any Python function that executes a machine learning model prediction and exposes
it through an HTTP/gRPC API. PyTriton installs Triton Inference Server in your environment and uses it for handling
HTTP/gRPC requests and responses. Our library provides a Python API that allows attaching a Python function to Triton
and a communication layer to send/receive data between Triton and the function. This solution helps utilize the
performance features of Triton Inference Server, such as dynamic batching or response cache, without changing your model
environment. Thus, it improves the performance of running inference on GPU for models implemented in Python. The solution is
framework-agnostic and can be used along with frameworks like PyTorch, TensorFlow, or JAX.

## Architecture

The diagram below presents the schema of how the Python models are served through Triton Inference Server using
PyTriton. The solution consists of two main components:

- Triton Inference Server: for exposing the HTTP/gRPC API and benefiting from performance features like dynamic batching
or response cache.
- Python Model Environment: your environment where the Python model is executed.


The Triton Inference Server binaries are provided as part of the PyTriton installation. The Triton Server is
installed in your current environment (system or container). The PyTriton controls the Triton Server process
through the `Triton Controller`.

Exposing the model through PyTriton requires the definition of an `Inference Callable` - a Python function that is
connected to Triton Inference Server and executes the model or ensemble for predictions. The integration layer binds
the `Inference Callable` to Triton Server and exposes it through the Triton HTTP/gRPC API under a provided `<model name>`. Once
the integration is done, the defined `Inference Callable` receives data sent to the HTTP/gRPC API endpoint
`v2/models/<model name>/infer`. Read more about HTTP/gRPC interface in Triton Inference Server
[documentation](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/inference_protocols.md#httprest-and-grpc-protocols).

The HTTP/gRPC requests sent to `v2/models/<model name>/infer` are handled by Triton
Inference Server. The server batches requests and passes them to the `Proxy Backend`, which sends the batched requests to the appropriate
`Inference Callable`. The data is sent as a `numpy` array. Once the `Inference Callable` finishes execution of
the model prediction, the result is returned to the `Proxy Backend`, and a response is created by Triton Server.

![High Level Design](assets/hld.svg)

## Serving the models

PyTriton provides an option to serve your Python model using Triton Inference Server to
handle HTTP/gRPC
requests and pass the input/output tensors to and from the model. We use a blocking mode where the application is a
long-lived process deployed in your cluster to serve the requests from clients.

Before you run the model for serving the inference callback function, it has to be defined. The inference callback receives the
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

The `infer_fn` receives the batched input data for the model and should return the batched outputs.

In the next step, you need to create a connection between Triton and the model. For that purpose, the `Triton` class has to
be used, and the `bind` method is required to be called to create a dedicated connection between Triton Inference
Server and the defined `infer_fn`.

In the blocking mode, we suggest using the `Triton` object as a context manager where multiple models can be loaded in
the way presented below:

<!--pytest-codeblocks:cont-->

```python {"skip": true}
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

At this point, you have defined how the model has to be handled by Triton and where the HTTP/gRPC requests for the model have
to be directed. The last part for serving the model is to call the `serve` method on the Triton object:


```python {"skip": true}
with Triton() as triton:
    # ...
    triton.serve()
```

When the `.serve()` method is called on the `Triton` object, the inference queries can be sent to
`localhost:8000/v2/models/MyModel`, and the `infer_fn` is called to handle the inference query.

## Working in the Jupyter Notebook

The package provides an option to work with your model inside the Jupyter Notebook. We call it a
background mode where
the model is deployed on Triton Inference Server for handling HTTP/gRPC requests, but there are other actions that you
want to perform after loading and starting serving the model.

Having the `infer_fn` defined in the same way as described in the [serving the models](#serving-the-models) section, you
can use the `Triton` object without a context:

```python
from pytriton.triton import Triton
triton = Triton()
```

In the next step, the model has to be loaded for serving in Triton Inference Server (which is also the same
as in the serving example):

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

Finally, to run the model in background mode, use the `run` method:


```python {"skip": true}
triton.run()
```

When the `.run()` method is called on the `Triton` object, the inference queries can be sent to
`localhost:8000/v2/models/MyModel`, and the `infer_fn` is called to handle the inference query.

The Triton server can be stopped at any time using the `stop` method:


```python {"skip": true}
triton.stop()
```

## What next?

Read more about using PyTriton in the [Quick Start](quick_start.md), [Examples](examples.md) and
find more options on how to configure Triton, models, and deployment on a cluster in the [Deploying Models](initialization.md)
section.

The details about classes and methods can be found in the [API Reference](api.md) page.
