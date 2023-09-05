..
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

PyTriton
==========

PyTriton is a Flask/FastAPI-like interface that simplifies Triton's deployment in Python environments.
The library allows serving Machine Learning models directly from Python through
NVIDIA's `Triton Inference Server`_.

.. _Triton Inference Server: https://github.com/triton-inference-server

In PyTriton, as in Flask or FastAPI, you can define any Python function that executes a machine learning model prediction and exposes
it through an HTTP/gRPC API. PyTriton installs Triton Inference Server in your environment and uses it for handling
HTTP/gRPC requests and responses. Our library provides a Python API that allows attaching a Python function to Triton
and a communication layer to send/receive data between Triton and the function. This solution helps utilize the
performance features of Triton Inference Server, such as dynamic batching or response cache, without changing your model
environment. Thus, it improves the performance of running inference on GPU for models implemented in Python. The solution is
framework-agnostic and can be used along with frameworks like PyTorch, TensorFlow, or JAX.


Installation
--------------

The package can be installed from `pypi`_ using:

.. _pypi: https://pypi.org/project/nvidia-pytriton/

.. code-block:: text

    pip install -U nvidia-pytriton

More details about installation can be found in the `documentation`_.

.. _documentation: https://triton-inference-server.github.io/pytriton/latest/installation/

Example
---------

The example presents how to run Python model in Triton Inference Server without need to change the current working
environment. In the example we are using a simple `Linear` PyTorch model.

The requirement for the example is to have installed PyTorch in your environment. You can do it running:


.. code-block:: text

    pip install torch

In the next step define the `Linear` model:

.. code-block:: python

    import torch

    model = torch.nn.Linear(2, 3).to("cuda").eval()

Create a function for handling inference request:

.. code-block:: python

    import numpy as np
    from pytriton.decorators import batch


    @batch
    def infer_fn(**inputs: np.ndarray):
        (input1_batch,) = inputs.values()
        input1_batch_tensor = torch.from_numpy(input1_batch).to("cuda")
        output1_batch_tensor = model(input1_batch_tensor)  # Calling the Python model inference
        output1_batch = output1_batch_tensor.cpu().detach().numpy()
        return [output1_batch]


In the next step, create the connection between the model and Triton Inference Server using the bind method:

.. code-block:: python

    from pytriton.model_config import ModelConfig, Tensor
    from pytriton.triton import Triton

    # Connecting inference callback with Triton Inference Server
    with Triton() as triton:
        # Load model into Triton Inference Server
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

Finally, serve the model with Triton Inference Server:

.. code-block:: python

    from pytriton.triton import Triton

    with Triton() as triton:
        ...  # Load models here
        triton.serve()

The `bind` method is creating a connection between Triton Inference Server and the `infer_fn` which handle
the inference queries. The `inputs` and `outputs` describe the model inputs and outputs that are exposed in
Triton. The config field allows more parameters for model deployment.

The `serve` method is blocking and at this point the application will wait for incoming HTTP/gRPC requests. From that
moment the model is available under name `Linear` in Triton server. The inference queries can be sent to
`localhost:8000/v2/models/Linear/infer` which are passed to the `infer_fn` function.

Links
-------

* Documentation: https://triton-inference-server.github.io/pytriton
* Source: https://github.com/triton-inference-server/pytriton
* Issues: https://github.com/triton-inference-server/pytriton/issues
* Changelog: https://github.com/triton-inference-server/pytriton/blob/main/CHANGELOG.md
* Known Issues: https://github.com/triton-inference-server/pytriton/blob/main/docs/known_issues.md
* Contributing: https://github.com/triton-inference-server/pytriton/blob/main/CONTRIBUTING.md
