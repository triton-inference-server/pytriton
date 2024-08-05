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

<!--pytest.mark.skip-->
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

At this point, you have defined how the model has to be handled by Triton and where the HTTP/gRPC requests for the model have
to be directed. The last part for serving the model is to call the `serve` method on the Triton object:


<!--pytest.mark.skip-->
```python
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

<!--pytest-codeblocks:cont-->
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

<!--pytest.mark.skip-->
```python
triton.run()
```

When the `.run()` method is called on the `Triton` object, the inference queries can be sent to
`localhost:8000/v2/models/MyModel`, and the `infer_fn` is called to handle the inference query.

The Triton server can be stopped at any time using the `stop` method:


<!--pytest.mark.skip-->
```python
triton.stop()
```

## In-depth Topics and Examples

### Model Deployment

Fine-tune your model deployment strategy with our targeted documentation:

- [Initialize Triton](initialization.md) for seamless startup.
- Bind your [models to Triton](binding_models.md) for enhanced communication.
- Adjust your [binding configurations](binding_configuration.md) for improved control.
- Expand your reach by [deploying on clusters](guides/deploying_in_clusters.md).
- Master the use of [Triton in remote mode](remote_triton.md).


### Inference Management

Hone your understanding of inference process management through PyTriton:

- Tailor the [Inference Callable](inference_callables/README.md) to your model's requirements.
- Use [decorators](inference_callables/decorators.md) to simplify your inference callbacks.
- Incorporate [custom parameters/headers](inference_callables/custom_params.md) for flexibility.
data.


### Dive into Examples

Visit the [examples directory](../examples/) for a curated selection of use cases ranging from basic to advanced, including:

- Standard model serving scenarios with different frameworks: PyTorch, TensorFlow2, JAX.
- Advanced setups like online learning, multi-node execution, or Kubernetes deployments.

### Troubleshooting

If you encounter any obstacles, our [Known Issues](known_issues.md) page is a helpful resource for troubleshooting common challenges.

### Streaming (alpha)

We introduced new alpha feature to PyTriton that allows to stream partial responses from a model. It is based on NVIDIA Triton Inference deocoupled models feature. Look at example in [examples/huggingface_dialogpt_streaming_pytorch](../examples/huggingface_dialogpt_streaming_pytorch).

### Profiling model

The [Perf Analyzer](https://github.com/triton-inference-server/perf_analyzer/blob/main/README.md) can be
used to profile models served through PyTriton. We have prepared an example of
using the Perf Analyzer to profile the BART PyTorch model. The example code can be found
in [examples/perf_analyzer](../examples/perf_analyzer).

Open Telemetry is a set of APIs, libraries, agents, and instrumentation to provide observability for cloud-native software. We have prepared an
[guide](guides/distributed_tracing.md) on how to use Open Telemetry with PyTriton.

## What next?

Read more about using PyTriton in the [Quick Start](quick_start.md), [Examples](../examples) and
find more options on how to configure Triton, models, and deployment on a cluster in the [Deploying Models](initialization.md)
section.

The details about classes and methods can be found in the [API Reference](reference/triton.md) page.

If there are any issues diffcult to invastigate, it is possible to use pytriton-check tool.
Usage is described in the [Basic Troubleshooting](guides/basic_troubleshooting.md) section.
