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

# Use custom parameters and headers

## Overview

The example presents a simple Add-Sub model which perform an addition and subtraction operations
on passed input data and scale them using parameters and http headers send to model by client.

Example consists of following scripts:

- `install.sh` - install additional dependencies
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
