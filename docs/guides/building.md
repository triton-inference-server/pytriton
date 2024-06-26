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
Additionally, it allows you to incorporate hotfixes that have not yet been officially released.

## Prerequisites

Before building the PyTriton binary package, ensure the following:

- Docker with [buildx plugin](https://github.com/docker/buildx) is installed on the system. For more information, refer to the Docker documentation.
- Access to the Docker daemon is available from the system or container.

If you plan to build `arm64` wheel on `amd64` machine we suggest to use QUEMU for emulation.
To enable QUEMU on Ubuntu you need to:
- Install the QEMU packages on your x86 machine:
```shell
sudo apt-get install qemu binfmt-support qemu-user-static
```
- Register the QEMU emulators for ARM architectures:
```shell
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
```

## Building PyTriton binary package

To build the wheel binary package, follow these steps from the root directory of the project:

```shell
make install-dev
make dist
```

*Note*: The default build create wheel for `x86_64` architecture. If you would like to build the wheel for `aarch64` use
```shell
make dist -e PLATFORM=linux/arm64
```
We use Docker convention name for platforms. The supported options are `linux/amd64` and `linux/arm64`.

The wheel package will be located in the `dist` directory. To install the library, run the following `pip` command:

```shell
pip install dist/nvidia_pytriton-*-py3-none-*.whl
```

*Note*: The wheel name would have `x86_64` or `aarch64` in name based on selected platform.

## Building for a specific Triton Inference Server version

Building for an unsupported OS or hardware platform is possible.
PyTriton requires a Python backend and either an HTTP or gRPC endpoint.
The build can be CPU-only, as inference is performed on Inference Handlers.

For more information on the Triton Inference Server build process, refer to the
[building section of Triton Inference Server documentation](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md).

!!! warning "Untested Build"

    The Triton Inference Server has only been rigorously tested on Ubuntu 22.04. Other OS and hardware platforms are not
    officially supported. You can test the build by following the steps outlined in the
    [Triton Inference Server testing guide](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/test.md).

By the following [docker method steps](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md#building-with-docker)
you can create a `tritonserver:latest` Docker image that can be used to build PyTriton with the following command:

```shell
make dist -e TRITONSERVER_IMAGE_VERSION=latest -e TRITONSERVER_IMAGE_NAME=tritonserver:latest
```
