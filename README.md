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

# PyTriton

The PyTriton is Flask/FastAPI-like interface to simplify Triton's deployment in Python environments.
The library allows to serve Machine Learning models directly from Python through
NVIDIA [Triton Inference Server](https://github.com/triton-inference-server).


<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [How it works?](#how-it-works)
- [Installation](#installation)
  - [Installing using pip](#installing-using-pip)
  - [Building binaries from source](#building-binaries-from-source)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Examples](#examples)
- [Profiling model](#profiling-model)
- [Documentation](#documentation)
- [Useful Links](#useful-links)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## How it works?

In PyTriton, same as in Flask or FastAPI, you can define any Python function that execute a Machine Learning model prediction and expose
it through HTTP/gRPC API. The library installs Triton Inference Server in your environment and use it for handling the
HTTP/gRPC requests and responses. Our library provides a Python API that allow to attach a Python function to the Triton
and a communication layer to send/receive data between Triton and the function. The solution helps use of the
performance features of Triton Inference Server, like dynamic batching or response cache, without changing your model
environment. Thus, improve the performance of running inference on GPU of models implemented in Python. The solution is
framework-agnostic and can be used along with frameworks like PyTorch, TensorFlow or JAX.

## Installation

The following prerequisites must be matched to perform an installation of library:

- Operating system with glibc >= 2.31. Triton Inference Server and PyTriton has only been rigorously tested on Ubuntu 20.04.
  Other supported operating systems include Ubuntu 20.04+, Debian 11+, Rocky Linux 9+, Red Hat Universal Base Image 9+.
- Python version >= 3.8. If you are using Python 3.9+, see the section "[Installation on Python 3.9+](docs/installation.md#installation-on-python-39)" for additional steps.
- pip >= 20.3

We assume you are comfortable with Python programming language
and familiar with Machine Learning models. Using [Docker](https://www.docker.com/) is an option, but not mandatory.

The library can be installed in:

- system environment
- virtualenv
- [Docker](https://www.docker.com/) image

The NVIDIA optimized Docker images for Python frameworks could be obtained
from [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/containers).

For using NVIDIA optimized Docker images we recommend to
install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html) to
run model inference on NVIDIA GPU.

### Installing using pip

The package can be installed from `pypi.org` using:

```shell
pip install -U pytriton
```

**Important**: The Triton Inference Server binary is installed as part of PyTriton package.

### Building binaries from source

The binary package can be built from the source, which enables flexibility to modify the PyTriton code
and integrate it with various versions of the Triton Inference Server, including custom builds.
For further information on building the PyTriton binary, refer to the [Building page](docs/building.md)

## Quick Start

The quick start presents how to run Python model in Triton Inference Server without need to change the current working
environment. In the example we are using a simple `Linear` PyTorch model.

The requirement for the example is to have installed PyTorch in your environment. You can do it running:

```shell
pip install torch
```

The integration of model requires to provide following elements:

- The model - framework or Python model or function that handle inference requests
- Inference callback - a lambda or function which handle the input data coming from Triton and return the result
- Python function connection with Triton Inference Server - a binding for communication between Triton and Python
  callback

In the next step define the `Linear` model:

```python
import torch

model = torch.nn.Linear(2, 3).to("cuda").eval()
```

In the second step create an inference callback as a function. The function as an argument obtain the HTTP/gRPC
request data in form of numpy array. The expected return object is also numpy array.

Example implementation:

<!--pytest-codeblocks:cont-->

```python
import numpy as np
from pytriton.decorators import batch

@batch
def infer_func(**inputs: np.ndarray):
    (input1_batch,) = inputs.values()
    input1_batch_tensor = torch.from_numpy(input1_batch).to("cuda")
    output1_batch_tensor = model(input1_batch_tensor)  # Calling the Python model inference
    output1_batch = output1_batch_tensor.cpu().detach().numpy()
    return [output1_batch]
```

In the next step, create the connection between the model and Triton Inference Server using the bind method:

<!--pytest-codeblocks:cont-->

```python
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

# Connecting inference callback with Triton Inference Server
with Triton() as triton:
    # Load model into Triton Inference Server
    triton.bind(
        model_name="Linear",
        infer_func=infer_func,
        inputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        outputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128)
    )
```

Finally, serve the model with Triton Inference Server:

<!--pytest.mark.skip-->

```python
from pytriton.triton import Triton

with Triton() as triton:
    ...  # Load models here
    triton.serve()
```

The `bind` method is creating a connection between Triton Inference Server and the `infer_func` which handle
the inference queries. The `inputs` and `outputs` describe the model inputs and outputs that are exposed in
Triton. The config field allow to provide more parameters for model deployment.

The `serve` method is blocking and at this point the application will wait for incoming HTTP/gRPC request. From that
point the model is available under name `Linear` in Triton server. The inference queries can be sent to
`localhost:8000/v2/models/Linear/infer` which are passed to the `infer_func` function.

If you would like to use the Triton in background mode use `run`. More about that you can find
in [documentation](https://triton-inference-server.github.io/pytriton).

Once the `server` or `run` method is called on `Triton` object the server status can be obtained using:

```shell
curl -v localhost:8000/v2/health/live
```

The model is loaded right after the server start and status can be queried using:

```shell
curl -v localhost:8000/v2/models/Linear/ready
```

Finally, you can send an inference query to the model:

```shell
curl -X POST \
  -H "Content-Type: application/json"  \
  -d @input.json \
  localhost:8000/v2/models/Linear/infer
```

The `input.json` with sample query:

```json
{
  "id": "0",
  "inputs": [
    {
      "name": "INPUT_1",
      "shape": [
        1,
        2
      ],
      "datatype": "FP32",
      "parameters": {},
      "data": [
        [
          -0.04281254857778549,
          0.6738349795341492
        ]
      ]
    }
  ]
}
```

Read more
about HTTP/gRPC interface in Triton Inference
Server [documentation](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/inference_protocols.md#httprest-and-grpc-protocols)
.

You can also validate the deployed model a simple client can be used to perform inference requests:

<!--pytest.mark.skip-->

```python
import torch
from pytriton.client import ModelClient

input1_data = torch.randn(128, 2).cpu().detach().numpy()

with ModelClient("localhost:8000", "Linear") as client:
    result_dict = client.infer_batch(input1_data)

print(result_dict)
```

The full example code can be found in [examples/linear_random_pytorch](examples/linear_random_pytorch).

More information about running the server and models can be found
in [documentation](https://triton-inference-server.github.io/pytriton).

## Architecture

The diagram below presents the schema how the Python models are served through Triton Inference Server using the
PyTriton. The solutions consist of two main components:

- Triton Inference Server - for exposing the HTTP/gRPC API, and benefits from performance features like dynamic batching
  or response cache
- Python Model Environment - your environment where Python model is executed

The Triton Inference Server binaries are provided as part of the PyTriton installation. The Triton Server is
installed in your current environment (system or container). The PyTriton controls the Triton Server process
through `Triton Controller`.

Exposing the model through PyTriton requires definition of `Inference Callable` - a Python function that is
connected to Triton Inference Server and execute model or ensemble for predictions. The integration layer bind
`Inference Callable` to Triton Server and expose it through Triton HTTP/gRPC API under a provided `<model name>`. Once
the integration is done, the defined `Inference Callable` receive data send to HTTP/gRPC API endpoint
`v2/models/<model name>/infer`. Read more about HTTP/gRPC interface in Triton Inference Server
[documentation](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/inference_protocols.md#httprest-and-grpc-protocols)
.

The HTTP/gRPC requests send to `v2/models/<model name>/infer` are handled by Triton
Inference Server. The server batch requests and pass to the `Proxy Backend` which send the batched requests to appropriate
`Inference Callable`. The data is sent as `numpy` array. Once the `Inference Callable` finish execution of
model prediction the result is returned to `Proxy Backend` and response is created by Triton Server.

![High Level Design](docs/assets/hld.svg)

## Examples

We provide simple examples how to integrate the PyTorch, TensorFlow2, JAX and simple Python models with Triton Inference
Server using PyTriton. Each example provide an instruction describing how

The list of available model examples:

- [Add-Sub Python model](examples/add_sub_python)
- [Add-Sub Python model Jupyter Notebook](examples/add_sub_notebook)
- [BART PyTorch from HuggingFace](examples/huggingface_bart_pytorch)
- [BERT JAX from HuggingFace](examples/huggingface_bert_jax)
- [Identity Python model](examples/identity_python)
- [Linear RAPIDS/CuPy model](examples/linear_cupy)
- [Linear RAPIDS/CuPy model Jupyter Notebook](examples/linear_cupy_notebook)
- [Linear PyTorch model](examples/identity_python)
- [Multi-Layer TensorFlow2](examples/mlp_random_tensorflow2)
- [Multi Instance deployment for Linear PyTorch model](examples/multi_instance_linear_pytorch)
- [Multi Model deployment for Python models](examples/multiple_models_python)
- [OPT JAX MultiNode from HuggingFace](examples/huggingface_opt_multinode_jax)
- [NeMo Megatron GPT model with multi-node support](examples/nemo_megatron_gpt_multinode)

## Profiling model

The [Perf Analyzer](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md) can be
used to profile the models served through pytriton. We have prepared the example of
using Perf Analyzer to profile BART PyTorch model. The example code can be found
in [examples/perf_analyzer](examples/perf_analyzer).

## Documentation

More information how to customize Triton Inference Server, loading models, deployment on cluster and the API reference
can be found in the [documentation](https://triton-inference-server.github.io/pytriton)

## Useful Links

* [Changelog](CHANGELOG.md)
* [Known Issues](docs/known_issues.md)
* [Contributing](CONTRIBUTING.md)
