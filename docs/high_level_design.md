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

# High-Level Design

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