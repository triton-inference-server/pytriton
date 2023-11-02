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

# Huggingface OPT JAX Multi-node Deployment

This example shows how to easily deploy JAX large language models in a multi-node environment using PyTriton. In this
tutorial we will be working with [HuggingFace OPT](https://huggingface.co/docs/transformers/model_doc/opt) with up to
530B parameters.

## Overview

To run JAX in multi-GPU and/or multi-node environment we are
using [jax.distributed](https://jax.readthedocs.io/en/latest/_autosummary/jax.distributed.initialize.html#jax.distributed.initialize)
and [jax.experimental.pjit](https://jax.readthedocs.io/en/latest/_modules/jax/experimental/pjit.html) modules. To learn
more about using `pjit` and `jax.distributed` for running multi-node models please visit JAX docs.

Example consists of following scripts:

- [server.py](server.py) - this file runs the Triton server (with `--rank 0`) or JAX worker (with `--host_idx`
  greater than 0) on each node. It contains the code that distributes the inputs from the server to the workers.
- [client.py](client.py) - example of a simple client that calls the server with a single sample.
- [opt_utils.py](opt_utils.py) - lower level code used by [server.py](server.py). In this file we define functions that
  create a sharding strategy, copy model parameters from the cpu into multiple devices and run inference.
- [modeling_flax_opt.py](modeling_flax_opt.py) - slightly
  modified [HuggingFace file](https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_flax_opt.py)
  with OPT model definition. The main difference is that in the HuggingFace repository the model is initialized with FP32
  weights even when the operations are in FP16. In our file we use FP16 for both storing parameters and performing
  operations.

And configurations:

- `kubernetes` - example Helm Charts for serving and test inference in Kubernetes cluster

Below you can find a list of available models:

| model name        | pretrained | source                                                  |
|-------------------|------------|---------------------------------------------------------|
| facebook/opt-125m | True       | [HuggingFace](https://huggingface.co/facebook/opt-125m) |
| facebook/opt-350m | True       | [HuggingFace](https://huggingface.co/facebook/opt-350m) |
| facebook/opt-1.3b | True       | [HuggingFace](https://huggingface.co/facebook/opt-1.3b) |
| facebook/opt-2.7b | True       | [HuggingFace](https://huggingface.co/facebook/opt-2.7b) |
| facebook/opt-6.7b | True       | [HuggingFace](https://huggingface.co/facebook/opt-6.7b) |
| facebook/opt-13b  | True       | [HuggingFace](https://huggingface.co/facebook/opt-13b)  |
| facebook/opt-30b  | True       | [HuggingFace](https://huggingface.co/facebook/opt-30b)  |
| facebook/opt-66b  | True       | [HuggingFace](https://huggingface.co/facebook/opt-66b)  |
| random/125M       | False      |                                                         |
| random/350M       | False      |                                                         |
| random/1.3B       | False      |                                                         |
| random/2.7B       | False      |                                                         |
| random/5B         | False      |                                                         |
| random/6.7B       | False      |                                                         |
| random/13B        | False      |                                                         |
| random/20B        | False      |                                                         |
| random/30B        | False      |                                                         |
| random/66B        | False      |                                                         |
| random/89B        | False      |                                                         |
| random/17B        | False      |                                                         |
| random/310B       | False      |                                                         |
| random/530B       | False      |                                                         |

## Running example locally

In this section we describe running the JAX on multi-GPU and/or multi-node environment where manual setup of environment
is done on each node.

### Prerequisites

Each node must meet following requirements:

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html).
- [NVIDIA Driver](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) based on
  chosen version of framework container

### Building the Docker image

The easiest way of running this example is inside a [nvcr.io](https://catalog.ngc.nvidia.com/containers) TensorFlow2
container. Example `Dockerfile` that can be used to run the server:

```Dockerfile
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:23.10-tf2-py3
FROM ${FROM_IMAGE_NAME}

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV NCCL_LAUNCH_MODE="PARALLEL"

WORKDIR /workdir

COPY install.sh .
RUN ./install.sh
RUN pip install <pytriton package>

COPY . .
```

On each node we have to build the image (or download it from a registry).

```bash
docker build -t jax-llm:latest .
```

### Serving the model

On each node run:

```bash
docker run --net host --rm --gpus all jax-llm python server.py \
  --head-url "<head node IP>:<port (e.g. 1234)>" \
  --number-of-nodes <number of nodes> \
  --rank <current node index, head node has index 0> \
  --model-name <model_name> \
  --number-of-gpus <number of GPUs per node>
```

The server expects two inputs:

- `input` - string array of shape (`batch_size`, 1),
- `output_length` - int64 array of shape (`batch-size`, 1).

It returns a sing output:

- `output` - string array of shape (`batch_size`, 1).

To read more about Triton server please
visit [Triton docs](https://github.com/triton-inference-server/server#documentation).

### Testing the inference

To use our example client run on any machine:

```bash
docker run --net host jax-llm python client.py \
  --server-url "http://<head node IP>:8000" \
  --input "<input text>" \
  --output-length <output length>
```

## Kubernetes example of running server on single/multiple nodes with multiple GPUs

This section describe how to server the JAX model on Kubernetes cluster. The following prerequisites must be matched to
run the example:

- Kubernetes cluster with NVIDIA GPU node
- [NVIDIA Device Plugin](https://github.com/NVIDIA/k8s-device-plugin) installed in Kubernetes cluster
- Docker Containers Registry accessible from Kubernetes cluster
- [Installed Helm](https://helm.sh/docs/intro/install/) for creating the deployment and test job

Optionally you may install NVIDIA Container Toolkit and NVIDIA GPU Operator which enable more features
like [MIG](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/gpu-operator-mig.html) or
[Time Slicing](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/gpu-sharing.html) support in the cluster.
To learn more how to set up Kubernetes cluster with NVIDIA GPU you can review [
NVIDIA Cloud Native Documentation](https://docs.nvidia.com/datacenter/cloud-native/contents.html)

### Deployment instruction

Below, we present a step-by-step guide assuming that **all the commands are executed from the root of repository**.

Follow these steps to run and test example in the cluster:

1. [Optional] Build PyTriton wheel following the [build instruction](../../docs/building.md)
2. Prepare the tag under which image is going to be pushed to your Docker Containers Registry accessible from Kubernetes
   cluster. Example for local cluster (minikube, k3s) with registry hosted inside the cluster:

```shell
export DOCKER_IMAGE_NAME_WITH_TAG=localhost:5000/jax-example:latest
```

3. Build and push the Docker container image to your registry:

```shell
# Export the base image used for build. We use TensorFlow image for JAX
export FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:23.10-tf2-py3
./examples/huggingface_opt_multinode_jax/kubernetes/build_and_push.sh
```
**Note**: By default the container is built using `pytriton` package from pypi.org. To build container with wheel built
locally use `export BUILD_FROM=dist` before executing script.

4. At this point there are 2 options to deploy the model depending on the size of the model:
   a) Install the Helm Chart with deployment and service for single-node:

```shell
helm upgrade -i --set deployment.image=${DOCKER_IMAGE_NAME_WITH_TAG} \
--set deployment.numOfGPUs=1 \
jax-example \
./examples/huggingface_opt_multinode_jax/kubernetes/single-node
```

b) Install the Helm Chart with deployment and service for multi-node:

**Important**: Running multi-node requires to create Persistent Volume Claim in the cluster shared between PODs. You can
pass name as argument to Helm Chart during installation. Read more how to create
[Persistent Volume Claim](#creating-persistent-volume-claim).

**Please note**: The multi-node deployment for scaling requires improved configuration of services and load balancing.

```shell
helm upgrade -i --set statefulset.image=${DOCKER_IMAGE_NAME_WITH_TAG} \
--set statefulset.persistentVolumeClaim=llm-cache-pvc \
--set statefulset.numOfNodes=3 \
--set statefulset.numOfGPUs=1 \
jax-example \
./examples/huggingface_opt_multinode_jax/kubernetes/multi-node
```

5. Install the Helm Chart with client test

```shell
helm install --set image=${DOCKER_IMAGE_NAME_WITH_TAG} \
jax-example-test \
./examples/huggingface_opt_multinode_jax/kubernetes/test
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
helm uninstall jax-example-test
helm uninstall jax-example
```

### Creating Persistent Volume Claim

This section describe how to create Persistent Volume Claim in Kuberenetes cluster using CSI or NFS drive.

#### Using CSI host path

When you are running on local machine (ex. Minikube or k3s) you can use CSI host path to create a persistent volume
claim. Make sure that appropriate extension for your cluster has been installed and run:

```shell
kubectl apply -f ./examples/huggingface_opt_multinode_jax/kubernetes/persistent-volume-claim-csi.yaml
```

#### Using NFS disk

When you are running Kubernetes cluster in Cloud Service Provider you can create persistent volume claim using NFS disk.

First, create the NFS disk and obtain its IP address. Make sure the disk is in the same network as Kubernetes cluster.
The pre-defined file share name for the NFS storage is `llm`.

Next modify the `./examples/huggingface_opt_multinode_jax/kubernetes/persistent-volume-claim-nfs.yaml` file and update the
`{IP}` value. Then run:

```shell
kubectl apply -f ./examples/huggingface_opt_multinode_jax/kubernetes/persistent-volume-nfs.yaml
```

Once the persistent volume is ready the claim can be created using:
```shell
kubectl apply -f ./examples/huggingface_opt_multinode_jax/kubernetes/persistent-volume-claim-nfs.yaml
```