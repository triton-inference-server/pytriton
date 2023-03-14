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

# Online learning example on MNIST dataset

## Overview

The example presents a simple Online Learning concept based on MNIST dataset.
There are two models: inference and training and both are deployed on Triton Inference Server.
Inference model is used for inference requests and training model is used for training.
Inference model is replaced with training model after each training epoch.

Example consists of following scripts:

- `install.sh` - install additional dependencies
- `server.py` - start the model with Triton Inference Server
- `client_infer.py` - execute HTTP/gRPC requests to the deployed model
- `client_train.py` - execute HTTP/gRPC requests to the deployed model for training
- `model.py` - model definition

## Quick Start

The step-by-step guide:

1. Install `pytriton` following the [installation instruction](../../README.md#installation)
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
6. Run the `client_infer.py` to perform inference requests (with test dataset) on the deployed model
and calculate accuracy. At the beginning accuracy should be around 10% (random predictions).
In the next steps you will run the training, so after a while accuracy should increase.

```shell
./client_infer.py
```

7. Open new terminal tab (ex. `Ctrl + T` on Ubuntu) or window
8. Go to the example directory
9. Run the `client_train.py` to perform training inference with training dataset on the training model
(The script communicates epoch number). The inference model will be replaced with training model after each epoch,
so in the ./client_infer.py terminal you should see the accuracy increasing.

```shell
./client_train.py
```
