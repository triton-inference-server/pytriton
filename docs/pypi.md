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

# Project description


The NVIDIA Pytriton is Flask/FastAPI-like interface to simplify Triton's deployment in Python environments.
The library allows to serve Machine Learning models directly from Python through
NVIDIA [Triton Inference Server](https://github.com/triton-inference-server).

In NVIDIA Pytriton, same as in Flask or FastAPI, you can define any Python function that execute a Machine Learning model prediction and expose
it through HTTP/gRPC API. The library installs Triton Inference Server in your environment and use it for handling the
HTTP/gRPC requests and responses. Our library provides a Python API that allow to attach a Python function to the Triton
and a communication layer to send/receive data between Triton and the function. The solution helps use of the
performance features of Triton Inference Server, like dynamic batching or response cache, without changing your model
environment. Thus, improve the performance of running inference on GPU of models implemented in Python. The solution is
framework-agnostic and can be used along with frameworks like PyTorch, TensorFlow or JAX.


# Installing

The package can be installed from `pypi.org` using:

```shell
pip install -U pytriton
```

# Example

The example presents how to run Python model in Triton Inference Server without need to change the current working
environment. In the example we are using a simple `Linear` PyTorch model.

The requirement for the example is to have installed PyTorch in your environment. You can do it running:

```shell
pip install torch
```

In the next step define the `Linear` model:

```python
import torch

model = torch.nn.Linear(2, 3).to("cuda").eval()
```

Create a function for handling inference request:

```python
import numpy as np

@batch
def infer_func(**inputs: np.ndarray):
    (input1_batch,) = inputs.values()
    input1_batch_tensor = torch.from_numpy(input1_batch).to("cuda")
    output1_batch_tensor = model(input1_batch_tensor)  # Calling the Python model inference
    output1_batch = output1_batch_tensor.cpu().detach().numpy()
    return [output1_batch]
```

Bind the function to Triton Inference Server and start server to handle HTTP/gRPC requests:
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
    # Serve model through Triton Inference Server
    triton.serve()
```

The `bind` method is creating a connection between Triton Inference Server and the `infer_func` which handle
the inference queries. The `inputs` and `outputs` describe the model inputs and outputs that are exposed in
Triton. The config field allow to provide more parameters for model deployment.

The `serve` method is blocking and at this point the application will wait for incoming HTTP/gRPC request. From that
point the model is available under name `Linear` in Triton server. The inference queries can be sent to
`localhost:8000/v2/models/Linear/infer` which are passed to the `infer_func` function.


# Links

* Documentation: https://triton-inference-server.github.io/pytriton
* Source: https://github.com/triton-inference-server/pytriton
* Issues: https://github.com/triton-inference-server/pytriton/issues
* Changelog: https://github.com/triton-inference-server/pytriton/CHANGELOG.md
* Known Issues: https://github.com/triton-inference-server/pytriton/docs/known_issues.md
* Contributing: https://github.com/triton-inference-server/pytriton/CONTRIBUTING.md
