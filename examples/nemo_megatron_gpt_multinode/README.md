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

# NeMo Megatron GPT model deployment

This example shows how to deploy NeMo Megatron GPT model using PyTriton.

## Introduction

[NVIDIA NeMo Megatron](https://developer.nvidia.com/nemo/megatron) is an end-to-end framework
for training and deploying LLMs with billions and trillions of parameters.

Example contains server and client prepared for text generation task.

## Requirements

Workstation on which you will start server script should meet requirements defined in model documentation.
Example documentation for this example default model - [NeMo Megatron GPT 1.3B](https://huggingface.co/nvidia/nemo-megatron-gpt-1.3B).

The easiest way to run this examples is to run it in [NVIDIA NeMo Docker container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo)
where environment is ready out-of-the box. Alternatively you can set up your environment manually as in [model instruction](https://huggingface.co/nvidia/nemo-megatron-gpt-1.3B#step-1-install-nemo-and-dependencies).

If you select to use container we recommend to install
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html).

## Running server

1. Run NVIDIA NeMo docker container:
    ```bash
    cd <pytriton_repository_dir>
    docker run \
        --rm -it \
        --gpus all --shm-size 2G \
        -v $PWD:$PWD -w $PWD \
        -p 8000:8000 -p 8001:8001 -p 8002:8002 \
        --name nemo_megatron_gpt_server \
    nvcr.io/nvidia/nemo:22.12 bash
    ```

    For more information on docker/cluster deployments see [documentation](../../docs/deploying_models.md#deploying-in-cluster).

2. While being in just started container, [install PyTriton](../../docs/installation.md):
    ```bash
    pip install -U pytriton
    ```

3. Start NeMo Megatron GPT model:
    ```bash
    ./examples/nemo_megatron_gpt_multinode/server.py
    ```

   By default, NeMo Megatron uses all available GPUs distributing model with increased data parallel size if possible.
   Tensor model parallel size (TP) and pipeline model parallel size (PP) stays the same as during training of the loaded model.
   For example for NeMo Megatron GPT 20B TP=4 and PP=1 inference on workstation with 8 GPUs will use 2 copies of model
   (data parallelism size = 2).

   To run NeMo Megatron GPT model inference on exact GPUs number just run in your python environment:

   ```bash
   ./examples/nemo_megatron_gpt_multinode/server.py --gpus 2   # uses first 2 GPUs
   ./examples/nemo_megatron_gpt_multinode/server.py --gpus 2,3   # uses 2 GPUs with ids 2 and 3
   ./examples/nemo_megatron_gpt_multinode/server.py --gpus -1  # uses all available GPUs
   ```

   For TP and PP model parameters refer to used model documentation.

   You can select NeMo Megatron repository on HuggingFace Hub with `--model-repo-id` parameter.
   ```bash
   ./examples/nemo_megatron_gpt_multinode/server.py --model-repo-id nvidia/nemo-megatron-gpt-20B
   ```

Server script will:
1. Ensure model and tokenizer data are downloaded
2. Load downloaded model and initialize environment for it
3. Start PyTriton server on listening on [configured ports](../../docs/deploying_models.md#configuring-triton).

   In this example HTTP endpoint will listen on 8000 port and thanks to
   [Docker port publishing](https://docs.docker.com/config/containers/container-networking/#published-ports)
   should be accessible outside of container.


## Slurm example of running server on multiple nodes with multiple GPUs

This example supports also scenario where model inference is performed in a multiple nodes, multiple gpus scenario.
For that we can use [Slurm](https://slurm.schedmd.com/) cluster management system.

1. Prepare Slurm submission file. Example Slurm `sbatch` submission file:

   ```shell
   #!/usr/bin/env bash

   #SBATCH --job-name=nemo_megatron_gpt
   #SBATCH --nodes=2
   #SBATCH --ntasks-per-node=8
   #SBATCH --open-mode=append
   #SBATCH --output=slurm_job-%x-%J.out
   #SBATCH --partition=<your_partition>
   #SBATCH --time=2:00:00

   set -x

   # assume that your current working directory is PyTriton repository
   # use github.com/nvidia/pyxis plugin
   srun --output slurm_job-%x-%J.out \
        --container-name nemo_megatron_gpt_container \
        --container-image nvcr.io/nvidia/nemo:22.12 \
        --container-mounts "${PWD}":"${PWD}" \
        --container-workdir "${PWD}" \
        --no-container-mount-home \
        --unbuffered \
        bash -c '[[ ${LOCAL_RANK} -eq 0 ]] && pip install -U pytriton || true'
   # reuse of container prepared in above srun
   srun --output slurm_job-%x-%J.out \
        --container-name nemo_megatron_gpt_container \
        --container-mounts "${PWD}":"${PWD}" \
        --container-workdir "${PWD}" \
        --no-container-mount-home \
        --unbuffered \
        bash -c "./examples/nemo_megatron_gpt_multinode/server.py --gpus 8 --nodes 2 --model-repo-id nvidia/nemo-megatron-gpt-20B"
   ```

   Parameters values requirements:
   - Number of tasks have to be equal to the product of `--nodes` and `--gpus`.
   - Number of nodes on which script is run have to be equal `--nodes` parameter.

   There might be need to add/modify `sbatch` and `srun` commands parameters - refer to documentation of your cluster for more information.

2. Submit job and observe logs in defined log paths:

   ```shell
   sbatch <submission_file>
   ```

## Running client

Client implements simple text generation task.

You can run client script in any python environment which have network access to NVIDIA Triton server endpoints.

Server script logs contains `Server http url http://<hostname>:<port>` line which defines hostname and port on which
Triton Inference Server http end-point is binded.
While running server in docker container url might contain docker container id in place of hostname. Such url
is unusable outside of container.

Example client calls:
- run it in same container as server:

   ```bash
   docker exec nemo_megatron_gpt_server ./examples/nemo_megatron_gpt_multinode/client.py --url http://localhost:8000
   ```

- run it in separate container:

   ```bash
   cd <pytriton_repository_dir>
   docker run --rm -it -v $PWD:$PWD -w $PWD --link nemo_megatron_gpt_server nvcr.io/nvidia/pytorch:22.12-py3 bash
   # now inside obtained container
   pip install -U pytriton
   ./examples/nemo_megatron_gpt_multinode/client.py --url http://nemo_megatron_gpt_server:8000
   ```

- run it locally:

   ```bash
   # setup python virtualenv if needed
   pip install virtualenv
   virtualenv -p $(which python3.8) .venv
   source .venv/bin/activate
   # and install pytriton
   pip install -U pytriton
   # run client
   # thanks to docker port publishing it is available outside of docker
   ./examples/nemo_megatron_gpt_multinode/client.py --url http://localhost:8000
   ```

For multi-node scenario you can run client:

- locally as in single node scenario with url pointing cluster workstation
- on cluster in same container as server:

   ```shell
   # attach to container
   srun --pty \
      --jobid <slurm_job_id_with_server> \
      --container-name <container_name_set_during_start_of_server> \
      --container-mounts "${PWD}:${PWD}" \
      --container-workdir "${PWD}" \
      --no-container-mount-home \
      bash
   # in just attached container
  ./examples/nemo_megatron_gpt_multinode/client.py --url http://<pytriton_hostname>:8000
   ```
- on cluster in new container:
   ```shell
  # start new container
   srun --pty \
      --partition <your_partiion> \
      --container-image nvcr.io/nvidia/nemo:22.12 \
      --container-mounts "${PWD}:${PWD}" \
      --container-workdir "${PWD}" \
      --no-container-mount-home \
      bash
   # in newly created container install PyTriton and execute
   pip install -U pytriton
   ./examples/nemo_megatron_gpt_multinode/client.py --url http://<pytriton_hostname>:8000
   ```

   There might be need to add/modify `srun` commands parameters - refer to documentation of your cluster for more information.

To prepare custom prompts use `--prompts` argument:

```shell
./examples/nemo_megatron_gpt_multinode/client.py --prompts "Thank you for" "Q: Are you going for a lunch?"
```

## Results

As a result client prints sequences generated by model based on prompts sent in request.

```
> ./examples/nemo_megatron_gpt_multinode/client.py --url http://nemo_megatron_gpt_server:8000 --prompts "Thank you for" "Q: Are you going for a lunch?"
================================
Thank you for the book. I have a question about Sissie in particular because we were supposed to watch Rocky. She didn't do anything like that, so now I'm wondering if she
================================
Q: Are you going for a lunch?
Lunch is typically served cold but I'm interested in ordering this dessert which, of course, doesn't come up as well on menus online.
```

## Kubernetes example of running server on single/multiple nodes with multiple GPUs

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

### Deployment instruction

Below, we present a step-by-step guide assuming that **all the commands are executed from the root of repository**.

Follow these steps to run and test example in the cluster:
1. [Optional] Build PyTriton wheel following the [build instruction](../../docs/building.md)
2. Prepare the tag under which image is going to be pushed to your Docker Containers Registry accessible from Kubernetes
cluster. Example for local cluster (minikube, k3s) with registry hosted inside the cluster:
```shell
export DOCKER_IMAGE_NAME_WITH_TAG=localhost:5000/nemo-example:latest
```
3. Build and push the Docker container image to your registry:

```shell
# Export the base image used for build
export FROM_IMAGE_NAME=nvcr.io/nvidia/nemo:22.12
./examples/nemo_megatron_gpt_multinode/kubernetes/build_and_push.sh
```

**Note**: By default the container is built using `pytriton` package from pypi.org. To build container with wheel built
locally use `export BUILD_FROM=dist` before executing script.

4. At this point there are 2 options to deploy the model depending on the size of the model:
a) Install the Helm Chart with deployment and service for single-node:
```shell
helm upgrade -i --set deployment.image=${DOCKER_IMAGE_NAME_WITH_TAG} \
--set deployment.numOfGPUs=1 \
nemo-example \
./examples/nemo_megatron_gpt_multinode/kubernetes/single-node
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
nemo-example \
./examples/nemo_megatron_gpt_multinode/kubernetes/multi-node
```

5. Install the Helm Chart with client test

```shell
helm install --set image=${DOCKER_IMAGE_NAME_WITH_TAG} \
nemo-example-test \
./examples/nemo_megatron_gpt_multinode/kubernetes/test
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
helm uninstall nemo-example-test
helm uninstall nemo-example
```

### Creating Persistent Volume Claim

This section describe how to create Persistent Volume Claim in Kuberenetes cluster using CSI or NFS drive.

#### Using CSI host path

When you are running on local machine (ex. Minikube or k3s) you can use CSI host path to create a persistent volume
claim. Make sure that appropriate extension for your cluster has been installed and run:

```shell
kubectl apply -f ./examples/nemo_megatron_gpt_multinode/kubernetes/persistent-volume-claim-csi.yaml
```

#### Using NFS disk

When you are running Kubernetes cluster in Cloud Service Provider you can create persistent volume claim using NFS disk.

First, create the NFS disk and obtain its IP address. Make sure the disk is in the same network as Kubernetes cluster.
The pre-defined file share name for the NFS storage is `llm`.

Next modify the `./examples/nemo_megatron_gpt_multinode/kubernetes/persistent-volume-claim-nfs.yaml` file and update the
`{IP}` value. Then run:

```shell
kubectl apply -f ./examples/nemo_megatron_gpt_multinode/kubernetes/persistent-volume-nfs.yaml
```

Once the persistent volume is ready the claim can be created using:
```shell
kubectl apply -f ./examples/nemo_megatron_gpt_multinode/kubernetes/persistent-volume-claim-nfs.yaml
```