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

# Custom HTTP/gRPC headers and parameters

This document provides guidelines for using custom HTTP/gRPC headers and parameters with PyTriton.
Original Triton documentation related to parameters can be found [here](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_parameters.md).
Now, undecorated inference function accepts list of Request instances.
Request class contains following fields:

- data - for inputs (stored as dictionary, but can be also accessed with request dict interface e.g. request["input_name"])
- parameters - for combined parameters and HTTP/gRPC headers

!!! warning "Parameters/headers usage limitations"

    Currently, custom parameters and headers can be only accessed in undecorated inference function (they don't work with decorators).
    There is separate example how to use parameters/headers in preprocessing step (see [here](../guides/downloaded_input_data.md))


## Parameters

Parameters are passed to the inference callable as a dictionary.
The dictionary is stored in HTTP/gRPC request body payload.

## HTTP/gRPC headers

Custom HTTP/gRPC headers are passed to the inference callable in the same dictionary as parameters,
but they are stored in HTTP/gRPC request headers instead of the request body payload.
For the headers it is also necessary to specify the header prefix in Triton config, which is used to distinguish  the custom
headers from standard ones (only headers with specified prefix are passed to the inference callable).

## Usage

1. Define inference callable (that one uses one parameter and one header):

    ```python
    import numpy as np
    from pytriton.model_config import ModelConfig, Tensor
    from pytriton.triton import Triton, TritonConfig

    def _infer_with_params_and_headers(requests):
        responses = []
        for req in requests:
            a_batch, b_batch = req.values()
            scaled_add_batch = (a_batch + b_batch) / float(req.parameters["header_divisor"])
            scaled_sub_batch = (a_batch - b_batch) * float(req.parameters["parameter_multiplier"])
            responses.append({"scaled_add": scaled_add_batch, "scaled_sub": scaled_sub_batch})
        return responses
    ```

2. Bind inference callable to Triton ("header" is the prefix for custom headers):

    <!--pytest-codeblocks:cont-->
    ```python
    triton = Triton(config=TritonConfig(http_header_forward_pattern="header.*"))
    triton.bind(
        model_name="ParamsAndHeaders",
        infer_func=_infer_with_params_and_headers,
        inputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        outputs=[
            Tensor(name="scaled_add", dtype=np.float32, shape=(-1,)),
            Tensor(name="scaled_sub", dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128),
    )

    triton.run()
    ```

3. Call the model using ModelClient:

    <!--pytest-codeblocks:cont-->

    ```python
    import numpy as np
    from pytriton.client import ModelClient

    batch_size = 2
    a_batch = np.ones((batch_size, 1), dtype=np.float32) * 2
    b_batch = np.ones((batch_size, 1), dtype=np.float32)
    ```
    <!--pytest-codeblocks:cont-->
    ```python
    with ModelClient("localhost", "ParamsAndHeaders") as client:
        result_batch = client.infer_batch(a_batch, b_batch, parameters={"parameter_multiplier": 2}, headers={"header_divisor": 3})
    ```


    <!--pytest-codeblocks:cont-->
    <!--
    This code is used by pytest to verify the correctness of the documentation.
    ```python
    triton.stop();

    assert np.allclose(result_batch["scaled_add"], (a_batch + b_batch) / 3)
    assert np.allclose(result_batch["scaled_sub"], (a_batch - b_batch) * 2)
    ```
    -->