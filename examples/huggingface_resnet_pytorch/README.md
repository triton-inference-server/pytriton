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

# HuggingFace ResNet50 PyTorch Model

## Overview

The example presents a HuggingFace ResNet50 PyTorch model inference.

Example consists of following scripts:

- `install.sh` - install additional dependencies for downloading model from HuggingFace
- `server.py` - start the model with Triton Inference Server
- `client.py` - execute HTTP/gRPC requests to the deployed model

And configurations:

- `kubernetes` - example Helm Charts for serving and test inference in Kubernetes cluster

## Running example locally

To run example locally the `torch` package is required. It can be installed in your current environment using pip:

```shell
pip install torch
```

Or you can use NVIDIA PyTorch container:

```shell
docker run -it --gpus 1 --shm-size 8gb -v {repository_path}:{repository_path} -w {repository_path} nvcr.io/nvidia/pytorch:24.02-py3 bash
```

If you select to use container we recommend to install
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html).

Follow the step-by-step guide to execute the example:

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

## Running example on Kubernetes cluster

The following prerequisites must be matched to run the example:

- Kubernetes cluster with NVIDIA GPU node
- [NVIDIA Device Plugin](https://github.com/NVIDIA/k8s-device-plugin) installed in Kubernetes cluster
- Docker Containers Registry accessible from Kubernetes cluster
- [Installed Helm](https://helm.sh/docs/intro/install/) for creating the deployment and test job

Optionally you may install NVIDIA Container Toolkit and NVIDIA GPU Operator which enable more features
like [MIG](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/gpu-operator-mig.html) or
[Time Slicing](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/gpu-sharing.html) support in the cluster.
To learn more how to set up Kubernetes cluster with NVIDIA GPU you can review [
NVIDIA Cloud Native Documentation](https://docs.nvidia.com/datacenter/cloud-native/contents.html)

Below, we present a step-by-step guide assuming that **all the commands are executed from the root of repository**.

Follow these steps to run and test example in the cluster:
1. [Optional] Build PyTriton wheel following the [build instruction](../../docs/building.md)
2. Prepare the tag under which image is going to be pushed to your Docker Containers Registry accessible from Kubernetes
cluster. Example for local cluster (minikube, k3s) with registry hosted inside the cluster:
```shell
export DOCKER_IMAGE_NAME_WITH_TAG=localhost:5000/resnet-pytorch-example:latest
```
3. Build and push the Docker container image to your registry:

```shell
# Export the base image used for build
export FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:24.02-py3
./examples/huggingface_resnet_pytorch/kubernetes/build_and_push.sh
```

**Note**: By default the container is built using `pytriton` package from `GitHub`. To build container with wheel built
locally use `export BUILD_FROM=dist` before executing script.

4. Install the Helm Chart with deployment and service:

```shell
helm upgrade -i --set deployment.image=${DOCKER_IMAGE_NAME_WITH_TAG} \
resnet-pytorch-example \
./examples/huggingface_resnet_pytorch/kubernetes/deployment
```

5. Install the Helm Chart with client test

```shell
helm install --set image=${DOCKER_IMAGE_NAME_WITH_TAG} \
resnet-pytorch-example-test \
./examples/huggingface_resnet_pytorch/kubernetes/test
```

Now, you can review the logs from the running PODs to verify the inference is running. To show the logs from cluster
for given POD first list all running pods:
```shell
kubectl get pods
```

Next show logs from server or client:
```shell
kubectl logs {NAME}
```

To remove the installed charts simply run:
```shell
helm uninstall resnet-pytorch-example-test
helm uninstall resnet-pytorch-example
```