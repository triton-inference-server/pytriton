<!--
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

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

# Building binary package from source

This guide provides an outline of the process for building the PyTriton binary package from source.
It offers the flexibility to modify the PyTriton code and integrate it with various versions
of the Triton Inference Server, including custom builds.

## Prerequisites

Before building the PyTriton binary package, ensure the following:

- Docker is installed on the system. For more information, refer to the Docker documentation.
- Access to the Docker daemon is available from the system or container.

## Building PyTriton binary package

To build the wheel binary package, follow these steps from the root directory of the project:

```shell
make install-dev
make dist
```

The wheel package will be located in the `dist` directory. To install the library, run the following `pip` command:

```shell
pip install dist/nvidia_pytriton-*-py3-none-*_x86_64.whl
```

## Building for a specific Triton Inference Server version

Building for an unsupported OS or hardware platform is possible.
PyTriton requires a Python backend and either an HTTP or gRPC endpoint.
The build can be CPU-only, as inference is performed on Inference Handlers.

For more information on the Triton Inference Server build process, refer to the
[building section of Triton Inference Server documentation](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md).

!!! warning "Untested Build"

    The Triton Inference Server has only been rigorously tested on Ubuntu 20.04. Other OS and hardware platforms are not
    officially supported. You can test the build by following the steps outlined in the
    [Triton Inference Server testing guide](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/test.md).

Using the following [docker method steps](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md#building-with-docker),
you can create a `tritonserver:latest` Docker image that can be used to build PyTriton with the following command:

By the following [docker method steps](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md#building-with-docker)
you can create a `tritonserver:latest` Docker image that can be used to build PyTriton with the following command:

```shell
make TRITONSERVER_IMAGE_NAME=tritonserver:latest dist
```