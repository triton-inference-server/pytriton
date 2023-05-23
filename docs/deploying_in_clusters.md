<!--
Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

# Deploying in Cluster

The library can be used inside containers and deployed on Kubernetes clusters. There are certain prerequisites and
information that would help deploy the library in your cluster.

## Health checks

The library uses the Triton Inference Server to handle HTTP/gRPC requests. Triton Server provides endpoints to validate if
the server is ready and in a healthy state. The following API endpoints can be used in your orchestrator to
control the application ready and live states:

- Ready: `/v2/health/ready`
- Live: `/v2/health/live`

## Exposing ports

The library uses the Triton Inference Server, which exposes the HTTP, gRPC, and metrics ports for communication. In the default
configuration, the following ports have to be exposed:

- 8000 for HTTP
- 8001 for gRPC
- 8002 for metrics

If the library is inside a Docker container, the ports can be exposed by passing an extra argument to the `docker run`
command. An example of passing ports configuration:

<!--pytest.mark.skip-->

```shell
docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 {image}
```

To deploy a container in Kubernetes, add a ports definition for the container in YAML deployment configuration:

```yaml
containers:
  - name: pytriton
    ...
    ports:
      - containerPort: 8000
        name: http
      - containerPort: 8001
        name: grpc
      - containerPort: 8002
        name: metrics
```

## Configuring shared memory

The connection between Python callbacks and the Triton Inference Server uses shared memory to pass data between the
processes. In the Docker container, the default amount of shared memory is 64MB, which may not be enough to pass input and
output data of the model. To increase the available shared memory size, pass an additional flag to the `docker run` command.
An example of increasing the shared memory size to 8GB:

<!--pytest.mark.skip-->

```shell
docker run --shm-size 8GB {image}
```
To increase the shared memory size for Kubernetes, the following configuration can be used:

```yaml
spec:
  volumes:
    - name: shared-memory
      emptyDir:
        medium: Memory
  containers:
    - name: pytriton
      ...
      volumeMounts:
        - mountPath: /dev/shm
          name: shared-memory
```

## Specify container init process

You can use the [`--init` flag](https://docs.docker.com/engine/reference/run/#specify-an-init-process) of the `docker run`
command to indicate that an init process should be used as the PID 1 in the container.
Specifying an init process ensures that reaping zombie processes are performed inside the container. The reaping zombie
processes functionality is important in case of an unexpected error occurrence in scripts hosting PyTriton.