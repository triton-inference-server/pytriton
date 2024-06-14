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

# HuggingFace BERT JAX Model

## Known issue

HuggingFace BERT JAX Model works only with containers 24.04 and newer due to the usage of new version of CUDA. Please make sure that you are using the correct container version.

## Overview

The example presents a HuggingFace BERT JAX model inference.

Example consists of following scripts:

- `install.sh` - install additional dependencies for downloading model from HuggingFace and JAX library
- `server.py` - start the model with Triton Inference Server
- `client.py` - execute HTTP/gRPC requests to the deployed model

## Quick Start

The step-by-step guide:

1. Install PyTriton following the [installation instruction](../../README.md#installation)
2. Install the additional packages using `install.sh`

```shell
./install.sh
```

3. In current terminal start the model on Triton using `server.py`

```shell
./server.py
```

4. Open new terminal tab (ex. `Ctrl + T` on Ubuntu) or window
5. Go to the example directory
6. Run the `client.py` to perform queries on model:

```shell
./client.py
```

