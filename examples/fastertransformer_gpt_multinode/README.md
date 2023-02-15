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

# FasterTransformer GPT model deployment

This example shows how to deploy the FasterTransformer GPT model using PyTriton.

## Introduction

[NVIDIA FasterTransformer](https://github.com/NVIDIA/FasterTransformer/) implements
a highly inference-optimized transformer-based encoder and decoder components.

The example contains a server and client prepared for a text generation task.

## Requirements

The workstation on which you will start a server script should meet the requirements defined in the FasterTransformer model to be deployed documentation.
Example documentation for this example default model - [FasterTransformer GPT](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md#requirements).

The easiest way to run this example is to run it on a docker image built with Dockerfile from
[FasterTransformer repository](https://github.com/NVIDIA/FasterTransformer). This docker image should contain
all required components to run an inference with FasterTransformer.

```shell
git clone https://github.com/NVIDIA/FasterTransformer
cd FasterTransformer
docker build -f docker/Dockerfile.torch -t nvidia/fastertransformer .
```

Alternatively, you can set up your environment manually as in [model instruction](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md#requirements).

## Running server

1. Start a docker container with NVIDIA FasterTransformer:
    ```bash
    cd <pytriton_repository_dir>
    docker run \
        -d --rm --gpus all --init --shm-size 32G \
        -e PYTHONPATH=/workspace/FasterTransformer \
        -e FT_REPO_DIR=/workspace/FasterTransformer \
        -e HF_HOME=${PWD}/hf \
        -v ${PWD}:${PWD} -w ${PWD} \
        -p 8000:8000 -p 8001:8001 -p 8002:8002 \
        --name ft_on_pytriton \
        nvidia/fastertransformer sleep 7d
    ```

    For more information on docker/cluster deployments, see [documentation](../../docs/deploying_models.md#deploying-in-cluster).

2. Install requirements required by example and [PyTriton](../../docs/installation.md):
    ```bash
    docker exec ft_on_pytriton bash -c "./examples/fastertransformer_gpt_multinode/install.sh && pip install -U pytriton"
    ```

3. Start a server with the FasterTransformer GPT model:
    ```bash
    docker exec ft_on_pytriton mpirun -n 1 --allow-run-as-root ./examples/fastertransformer_gpt_multinode/server.py
    ```

   FasterTransformer PyTorch Op which is used in this example uses MPI to organize all GPUs.
   A single GPU is assigned per MPI job.

   The inference tensor model parallel size (TP) and pipeline model parallel size (PP) parameters might be different
   from training ones.
   User should remember to start as many MPI jobs equal to the product of PP and TP parameters.
   To run FasterTransformer model inference on the 8 GPUs with TP=4 and PP=2 use:

   ```bash
   docker exec ft_on_pytriton \
        mpirun -n 8 --allow-run-as-root ./examples/fastertransformer_gpt_multinode/server.py --tp 4 --pp 2
   ```

   You can select the source model from HuggingFace Hub with `--hf-model-name` parameter:
   ```bash
   docker exec ft_on_pytriton \
        mpirun -n 8 --allow-run-as-root ./examples/fastertransformer_gpt_multinode/server.py \
            --model-repo-id nvidia/nemo-megatron-gpt-20B \
            --tp 8
   ```

Server script will:
1. Downloads source model and tokenizer data
2. Convert the source model to FasterTransformer format
3. Load a FasterTransformer model and initialize an environment for it
4. Start the PyTriton server listening on [configured ports](../../docs/deploying_models.md#configuring-triton).

   In this example, the HTTP endpoint will listen on the 8000 port, and thanks to
   [Docker port publishing](https://docs.docker.com/config/containers/container-networking/#published-ports)
   should be accessible outside of the container.


### Run server on multiple nodes with multiple GPUs

[//]: # (add info on )

This example supports the scenario where model inference is performed on multiple GPUs from multiple nodes.
For that, we can use [Slurm](https://slurm.schedmd.com/) cluster management system.

1. Prepare the slurm submission file. Example Slurm `sbatch` submission file:

   ```shell
   #!/usr/bin/env bash

   #SBATCH --job-name=ft_on_pytriton
   #SBATCH --nodes=2
   #SBATCH --ntasks-per-node=8
   #SBATCH --open-mode=append
   #SBATCH --output=slurm_job-%x-%J.out
   #SBATCH --partition=<your_partition>
   #SBATCH --time=2:00:00

   set -x
   export FT_REPO_DIR=/workspace/FasterTransformer
   export PYTHONPATH=/workspace/FasterTransformer
   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   export HF_HOME=${PWD}/hf

   # assume that your current working directory is the pytriton repository
   # use github.com/nvidia/pyxis plugin
   srun --output slurm_job-%x-%J.out \
        --container-name ft_on_pytriton \
        --container-image <docker_image_on_your_docker_registry> \
        --container-mounts "${PWD}":"${PWD}" \
        --container-workdir "${PWD}" \
        --no-container-mount-home \
        --unbuffered \
        bash -c '[[ ${LOCAL_RANK} -eq 0 ]] && (./examples/fastertransformer_gpt_multinode/install.sh && pip install -U pytriton) || true'
   # reuse of container prepared in above srun
   srun --output slurm_job-%x-%J.out \
        --container-name ft_on_pytriton \
        --container-mounts "${PWD}":"${PWD}" \
        --container-workdir "${PWD}" \
        --export FT_REPO_DIR,PYTHONPATH,CUDA_VISIBLE_DEVICES,HF_HOME \
        --no-container-mount-home \
        --unbuffered \
        bash -c "./examples/fastertransformer_gpt_multinode/server.py --tp 8 --pp 2 --hf-model-name gpt2"
   ```

   Parameters values requirements:
   - the number of tasks has to be equal to the product of `--nodes` and `--gpus` and product of `--tp` and `--pp` parameters.

   There might be a need to add/modify `sbatch` and `srun` commands parameters - refer to the documentation of your cluster for more information.

2. Submit the job and observe logs in defined log paths:

   ```shell
   sbatch <submission_file>
   ```

## Running client

Client implements simple text generation task.

You can run client script in any python environment having a network route to NVIDIA Triton server endpoints.

Example client calls:
- run it in the same container as the server:

   ```bash
   docker exec ft_on_pytriton ./examples/fastertransformer_gpt_multinode/client.py
   ```

- run it in a separate container:

   ```bash
   cd <pytriton_repository_dir>
   docker run --rm -it -v $PWD:$PWD -w $PWD --link fastertransformer_with_ft python:3 bash
   # now inside the obtained container
   pip install -U pytriton
   ./examples/fastertransformer_gpt_multinode/client.py --url http://fastertransformer_with_ft:8000
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
   # thanks to docker port publishing it is available outside of the docker
   ./examples/fastertransformer_gpt_multinode/client.py --url http://localhost:8000
   ```

For multi-node scenarios you can run the client:

- locally as in a single node scenario with URL pointing cluster workstation
- on a cluster in the same container as the server:

   ```shell
   # attach to the container
   srun --pty \
      --jobid <slurm_job_id_with_server> \
      --container-name ft_on_pytriton \
      --container-mounts "${PWD}:${PWD}" \
      --container-workdir "${PWD}" \
      --no-container-mount-home \
      bash
   # in just attached container
  ./examples/fastertransformer_gpt_multinode/client.py --url http://<pytriton_hostname>:8000
   ```
- on a cluster in a new container:
   ```shell
  # start a new container
   srun --pty \
      --partition <your_partiion> \
      --container-image python:3 \
      --container-mounts "${PWD}:${PWD}" \
      --container-workdir "${PWD}" \
      --no-container-mount-home \
      bash
   # in the newly created container install pytriton and execute
   pip install -U pytriton
   ./examples/fastertransformer_gpt_multinode/client.py --url http://<pytriton_hostname>:8000
   ```

   There might be a need to add/modify `srun` command parameters - refer to the documentation of your cluster for more information.

To prepare custom prompts use the `--prompts` argument:

```shell
./examples/fastertransformer_gpt_multinode/client.py --prompts "Thank you for " "Q: Are you going for lunch?"
```

## Results

As a result client prints sequences generated by the model based on prompts sent in the request.

```
> ./examples/fastertransformer_gpt_multinode/client.py --url http://ft_on_pytriton:8000 --prompts "Thank you for" "Q: Are you going for a lunch?"
================================
Thank you for the book. I have a question about Sissie in particular because we were supposed to watch Rocky. She didn't do anything like that, so now I'm wondering if she
================================
Q: Are you going for lunch?
Lunch is typically served cold but I'm interested in ordering this dessert which, of course, doesn't come up as well on menus online.
```
