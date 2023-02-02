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

# Quick Start

The prerequisite for this section is installing the NVIDIA PyTriton which can be found
in [installation](installation.md)
section.

The quick start presents how to run Python model in Triton Inference Server without need to change the current working
environment. In the example we are using a simple `Linear` PyTorch model.

The integration of model requires to provide following elements:

- The model - framework or Python model or function that handle inference requests
- Inference callback - a lambda or function which handle the input data coming from Triton and return the result
- Python function connection with Triton Inference Server - a binding for communication between Triton and Python callback

The requirement for the example is to have installed PyTorch in your environment. You can do it running:

<!--pytest.mark.skip-->

```shell
pip install torch
```

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
import torch

from pytriton.decorators import batch


@batch
def infer_fn(**inputs: np.ndarray):
    (input1_batch,) = inputs.values()
    input1_batch_tensor = torch.from_numpy(input1_batch).to("cuda")
    output1_batch_tensor = model(input1_batch_tensor) # Calling the Python model inference
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
    triton.bind(
        model_name="Linear",
        infer_func=infer_fn,
        inputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        outputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128)
    )
    ...
```

Finally, serve the model with Triton Inference Server:

<!--pytest.mark.skip-->

```python
from pytriton.triton import Triton

with Triton() as triton:
    ...  # Load models here
    triton.serve()
```

The `bind` method is creating a connection between Triton Inference Server and the `infer_fn` which handle
the inference queries. The `inputs` and `outputs` describe the model inputs and outputs that are exposed in
Triton. The config field allow to provide more parameters for model deployment.

The `serve` method is blocking and at this point the application will wait for incoming HTTP/gRPC request. From that
point the model is available under name `Linear` in Triton server.  The inference queries can be sent to
`localhost:8000/v2/models/Linear/infer` which are passed to the `infer_fn` function.

If you would like to use the Triton in background mode use `run`. More about that you can find
in [deploying models](deploying_models.md) section.

Once the `server` or `run` method is called on `Triton` object the server status can be obtained using:

<!--pytest.mark.skip-->

```shell
curl -v localhost:8000/v2/health/live
```

The model is loaded right after the server start and status can be queried using:

<!--pytest.mark.skip-->
```shell
curl -v localhost:8000/v2/models/Linear/ready
```

Finally, you can send an inference query to the model:
<!--pytest.mark.skip-->

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
      "shape": [1, 2],
      "datatype": "FP32",
      "parameters": {},
      "data": [[-0.04281254857778549, 0.6738349795341492]]
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

The full example code can be found in [examples/linear_random_pytorch](../examples/linear_random_pytorch).

More information about running the server and models can be found in [deploying models](deploying_models.md) section.