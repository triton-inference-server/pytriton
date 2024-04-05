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

# Linear model in PyTorch

## Overview

The example presents a simple Linear model implemented in PyTorch

Example consists of following scripts:

- `server.py` - start the model with Triton Inference Server
- `client.py` - execute HTTP/gRPC requests to the deployed model

## Requirements

The example requires the `torch` package. It can be installed in your current environment using pip:

```shell
pip install torch
```

Or you can use NVIDIA PyTorch container:
```shell
docker run -it --gpus 1 --shm-size 8gb -v {repository_path}:{repository_path} -w {repository_path} nvcr.io/nvidia/pytorch:24.03-py3 bash
```

If you select to use container we recommend to install
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html).

## Quick Start

The step-by-step guide:

1. Install PyTriton following the [installation instruction](../../README.md#installation)
2. In current terminal start the model on Triton using `server.py`

```shell
./server.py
```

3. Open new terminal tab (ex. `Ctrl + T` on Ubuntu) or window
4. Go to the example directory
5. Run the `client.py` to perform queries on model:

```shell
./client.py
```

