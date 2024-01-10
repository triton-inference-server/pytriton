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

Welcome to PyTriton, a Flask/FastAPI-like framework designed to streamline the use of NVIDIA's [Triton Inference Server](https://github.com/triton-inference-server) within Python environments. PyTriton enables serving Machine Learning models with ease, supporting direct deployment from Python.

For comprehensive guidance on how to deploy your models, optimize performance, and explore the API, delve into the extensive resources found in our [documentation](https://triton-inference-server.github.io/pytriton).

## Features at a Glance

The distinct capabilities of PyTriton are summarized in the feature matrix:

| Feature | Description |
| ------- | ----------- |
| Native Python support | You can create any [Python function](https://triton-inference-server.github.io/pytriton/latest/inference_callables/) and expose it as an HTTP/gRPC API. |
| Framework-agnostic | You can run any Python code with any framework of your choice, such as: PyTorch, TensorFlow, or JAX. |
| Performance optimization | You can benefit from [dynamic batching](https://triton-inference-server.github.io/pytriton/latest/inference_callables/decorators/#batch), response cache, model pipelining, [clusters](https://triton-inference-server.github.io/pytriton/latest/guides/deploying_in_clusters/), and GPU/CPU inference. |
| Decorators | You can use batching [decorators](https://triton-inference-server.github.io/pytriton/latest/inference_callables/decorators/) to handle batching and other pre-processing tasks for your inference function. |
| Easy [installation](https://triton-inference-server.github.io/pytriton/latest/installation/) and setup | You can use a simple and familiar interface based on Flask/FastAPI for easy installation and [setup](https://triton-inference-server.github.io/pytriton/latest/binding_models/).  |
| [Model clients](https://triton-inference-server.github.io/pytriton/latest/clients)   | You can access high-level model clients for HTTP/gRPC requests with configurable options and both synchronous and [asynchronous](https://triton-inference-server.github.io/pytriton/latest/clients/#asynciomodelclient)  API. |
| Streaming (alpha) | You can stream partial responses from a model by serving it in a [decoupled mode](https://triton-inference-server.github.io/pytriton/latest/clients/#decoupledmodelclient). |

Learn more about PyTriton's [architecture](https://triton-inference-server.github.io/pytriton/latest/high_level_design/).


## Prerequisites

Before proceeding with the installation of PyTriton, ensure your system meets the following criteria:

- **Operating System**: Compatible with glibc version `2.35` or higher.
  - Primarily tested on Ubuntu 22.04.
  - Other supported OS include Debian 11+, Rocky Linux 9+, and Red Hat UBI 9+.
  - Use `ldd --version` to verify your glibc version.
- **Python**: Version `3.8` or newer.
- **pip**: Version `20.3` or newer.
- **libpython**: Ensure `libpython3.*.so` is installed, corresponding to your Python version.


## Install

The PyTriton can be installed from [pypi.org](https://pypi.org/project/nvidia-pytriton/) by running the following command:

```shell
pip install nvidia-pytriton
```

**Important**: The Triton Inference Server binary is installed as part of the PyTriton package.

Discover more about PyTriton's [installation procedures](https://triton-inference-server.github.io/pytriton/latest/installation/), including Docker usage, prerequisites, and insights into [building binaries from source](https://triton-inference-server.github.io/pytriton/latest/guides/building/) to match your specific Triton server versions.


## Quick Start

The quick start presents how to run Python model in Triton Inference Server without need to change the current working
environment. In the example we are using a simple `Linear` model.

The `infer_fn` is a function that takes an `data` tensor and returns a list with single output tensor. The `@batch` from [batching decorators](https://triton-inference-server.github.io/pytriton/latest/inference_callables/decorators/) is used to handle batching for the model.

```python
import numpy as np
from pytriton.decorators import batch

@batch
def infer_fn(data):
    result = data * np.array([[-1]], dtype=np.float32)  # Process inputs and produce result
    return [result]
```


In the next step, you can create the binding between the inference callable and Triton Inference Server using the `bind` method from pyTriton. This method takes the model name, the inference callable, the inputs and outputs tensors, and an optional model configuration object.

<!--pytest-codeblocks:cont-->

```python
from pytriton.model_config import Tensor
from pytriton.triton import Triton
triton = Triton()
triton.bind(
    model_name="Linear",
    infer_func=infer_fn,
    inputs=[Tensor(name="data", dtype=np.float32, shape=(-1,)),],
    outputs=[Tensor(name="result", dtype=np.float32, shape=(-1,)),],
)
triton.run()
```

Finally, you can send an inference query to the model using the `ModelClient` class. The `infer_sample` method takes the input data as a numpy array and returns the output data as a numpy array. You can learn more about the `ModelClient` class in the [clients](https://triton-inference-server.github.io/pytriton/latest/clients/) section.

<!--pytest-codeblocks:cont-->

```python
from pytriton.client import ModelClient

client = ModelClient("localhost", "Linear")
data = np.array([1, 2, ], dtype=np.float32)
print(client.infer_sample(data=data))
```
After the inference is done, you can stop the Triton Inference Server and close the client:

<!--pytest-codeblocks:cont-->

```python
client.close()
triton.stop()
```

The output of the inference should be:

<!--pytest.mark.skip-->
```python
{'result': array([-1., -2.], dtype=float32)}
```


For the full example, including defining the model and binding it to the Triton server, check out our detailed [Quick Start](https://triton-inference-server.github.io/pytriton/latest/quick_start/) instructions. Get your model up and running, explore how to serve it, and learn how to [invoke it from client applications](https://triton-inference-server.github.io/pytriton/latest/clients/).


The full example code can be found in [examples/linear_random_pytorch](examples/linear_random_pytorch).

## Examples

The [examples](examples) page presents various cases of serving models using PyTriton. You can find simple examples of running PyTorch, TensorFlow2, JAX, and simple Python models. Additionally, we have prepared more advanced scenarios like online learning, multi-node models, or deployment on Kubernetes using PyTriton. Each example contains instructions describing how to build and run the example. Learn more about how to use PyTriton by reviewing our examples.


## Useful Links

- [Changelog](https://triton-inference-server.github.io/pytriton/latest/CHANGELOG/)
- [Version Management](https://triton-inference-server.github.io/pytriton/latest/CONTRIBUTING/#version-management)
- [Contributing](https://triton-inference-server.github.io/pytriton/latest/CONTRIBUTING/)
- [Known Issues](https://triton-inference-server.github.io/pytriton/latest/known_issues/)
