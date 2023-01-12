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

# Installation

This section describe how to install the library. We assume you are comfortable with Python programming language
and familiar with Machine Learning models. Using [Docker](https://www.docker.com/) is an option, but not mandatory.

## Prerequisites

The following prerequisites must be matched to perform an installation of library:

- Ubuntu 20.04 - required for Triton Inference Server binary compatibility
- Python version >= 3.8
- pip >= 20.3

The library can be installed in:

- system environment
- virtualenv
- [Docker](https://www.docker.com/) image based on `ubuntu:20.04`

The NVIDIA optimized Docker images for Python frameworks could be obtained
from [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/containers).

For using NVIDIA optimized Docker images we recommend to
install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html) to
run model inference on NVIDIA GPU.

## Installing using PIP

The package can be installed from `pypi.org` using:

```shell
pip install -U pytriton
```

## Building from source

The package can be also build from the source using `Make` commands run from the main project directory. The build
process requires Docker installed in your system. The instruction can be found
in [Docker documentation](https://docs.docker.com/engine/install/ubuntu/).

To prepare the wheel you need first install additional packages using:

```shell
make install-dev
```

Next run the build process:

```
make dist
```

The wheel would be located in `dist` catalog. Use pip command to install the library:

```shell
pip install dist/pytriton-*-py3-none-linux_x86_64.whl
```
