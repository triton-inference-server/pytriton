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

# Deploying Models

The following page provides more details about possible options for configuring the
[Triton Inference Server](https://github.com/triton-inference-server/server),
configuring the model for loading in Triton, and deploying the solution in Docker containers or clusters.

## Examples

Before you move to more advanced topics, you may want to review examples that provide an implementation of various
models (in JAX, Python, PyTorch, and TensorFlow) deployed using the library.

You can also find the usage
of [Perf Analyzer](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md)
for profiling models (throughput, latency) once deployed using the solution.

For more information, please review the [Examples](examples.md) page.

## Configuring Triton

The [Triton][pytriton.triton.Triton] class is the base entry point for working with Triton Inference Server.

### Initialization

Connecting Python models with Triton Inference Server working in the current environment requires creating
a [Triton][pytriton.triton.Triton] object. This can be done by creating a context:

```python
from pytriton.triton import Triton

with Triton() as triton:
    ...
```

or simply creating an object:

```python
from pytriton.triton import Triton

triton = Triton()
```

The Triton Inference Server behavior can be configured by passing [config][pytriton.triton.TritonConfig] parameter:

```python
import pathlib
from pytriton.triton import Triton, TritonConfig

triton_config = TritonConfig(log_file=pathlib.Path("/tmp/triton.log"))
with Triton(config=triton_config) as triton:
    ...
```

and through environment variables, for example, set as in the command below:

<!--pytest.mark.skip-->

```sh
PYTRITON_TRITON_CONFIG_LOG_VERBOSITY=4 python my_script.py
```

The order of precedence of configuration methods is:

- config defined through `config` parameter of [Triton][pytriton.triton.Triton] class `__init__` method
- config defined in environment variables
- default [TritonConfig][pytriton.triton.TritonConfig] values

### Blocking mode

The blocking mode will stop the execution of the current thread and wait for incoming HTTP/gRPC requests for inference
execution. This mode makes your application behave as a pure server. The example of using blocking mode:

<!--pytest.mark.skip-->

```python
from pytriton.triton import Triton

with Triton() as triton:
    ...  # Load models here
    triton.serve()
```

### Background mode

The background mode runs Triton as a subprocess and does not block the execution of the current thread. In this mode, you can run
Triton Inference Server and interact with it from the current context. The example of using background mode:

<!--pytest.mark.skip-->

```python
from pytriton.triton import Triton

triton = Triton()
...  # Load models here
triton.run()  # Triton Server started
print("This print will appear")
triton.stop()  # Triton Server stopped
```

## Loading models

The Triton class provides methods to load one or multiple models to the Triton server:

```python
import numpy as np
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton


@batch
def infer_fn(**inputs: np.ndarray):
    input1, input2 = inputs.values()
    outputs = model(input1, input2)
    return [outputs]

with Triton() as triton:
  triton.bind(
      model_name="ModelName",
      infer_func=infer_fn,
      inputs=[
          Tensor(shape=(1,), dtype=np.bytes_),  # sample containing single bytes value
          Tensor(shape=(-1,), dtype=np.bytes_)  # sample containing vector of bytes
      ],
      outputs=[
          Tensor(shape=(-1,), dtype=np.float32),
      ],
      config=ModelConfig(max_batch_size=8)
  )
```

The `bind` method's mandatory arguments are:

- `model_name`: defines under which name the model is available in Triton Inference Server
- `infer_func`: function or Python `Callable` object which obtains the data passed in the request and returns the output
- `inputs`: defines the number, types, and shapes for model inputs
- `outputs`: defines the number, types, and shapes for model outputs
- `config`: more customization for model deployment and behavior on the Triton server

Once the `bind` method is called, the model is created in the Triton Inference Server model store under
the provided `model_name`.

### Inference Callable

The inference callable is an entry point for inference. This can be any callable that receives the data for
model inputs in the form of a list of request dictionaries where input names are mapped into ndarrays.
Input can be also adapted to different more convenient forms using a set of decorators.
**More details about designing inference callable and using of decorators can be found
in [Inference Callable Design](inference_callable.md) page.**

In the simplest implementation for functionality that passes input data on output, a lambda can be used:

```python
import numpy as np
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

with Triton() as triton:
  triton.bind(
      model_name="Identity",
      infer_func=lambda requests: requests,
      inputs=[Tensor(dtype=np.float32, shape=(1,))],
      outputs=[Tensor(dtype=np.float32, shape=(1,))],
      config=ModelConfig(max_batch_size=8)
  )
```

### Multi-instance model inference

Multi-instance model inference is a mechanism for loading multiple instances of the same model and calling
them alternately (to hide transfer overhead).

With the `Triton` class, it can be realized by providing the list of multiple inference callables to `Triton.bind`
in the `infer_func` parameter.

The example presents multiple instances of the Linear PyTorch model loaded on separate devices.

First, define the wrapper class for the inference handler. The class initialization receives a model and device as
arguments. The inference handling is done by method `__call__` where the `model` instance is called:

```python
import torch
from pytriton.decorators import batch


class _InferFuncWrapper:
    def __init__(self, model: torch.nn.Module, device: str):
        self._model = model
        self._device = device

    @batch
    def __call__(self, **inputs):
        (input1_batch,) = inputs.values()
        input1_batch_tensor = torch.from_numpy(input1_batch).to(self._device)
        output1_batch_tensor = self._model(input1_batch_tensor)
        output1_batch = output1_batch_tensor.cpu().detach().numpy()
        return [output1_batch]
```

Next, create a factory function where a model and instances of `_InferFuncWrapper` are created - one per each device:

<!--pytest-codeblocks:cont-->

```python
def _infer_function_factory(devices):
    infer_fns = []
    for device in devices:
        model = torch.nn.Linear(20, 30).to(device).eval()
        infer_fns.append(_InferFuncWrapper(model=model, device=device))

    return infer_fns
```

Finally, the list of callable objects is passed to `infer_func` parameter of the `Triton.bind` function:

<!--pytest-codeblocks:cont-->

```python
import numpy as np
from pytriton.triton import Triton
from pytriton.model_config import ModelConfig, Tensor

with Triton() as triton:
  triton.bind(
      model_name="Linear",
      infer_func=_infer_function_factory(devices=["cuda", "cpu"]),
      inputs=[
          Tensor(dtype=np.float32, shape=(-1,)),
      ],
      outputs=[
          Tensor(dtype=np.float32, shape=(-1,)),
      ],
      config=ModelConfig(max_batch_size=16),
  )
  ...
```

Once the multiple callable objects are passed to `infer_func`, the Triton server gets information that multiple instances
of the same model have been created. The incoming requests are distributed among created instances. In our case executing
two instances of a `Linear` model loaded on CPU and GPU devices.

### Inputs and Outputs

The integration of the Python model requires the inputs and outputs types of the model. This is required to
correctly map the input and output data passed through the Triton Inference Server.

The simplest definition of model inputs and outputs expects providing the type of data and the shape per input:

```python
import numpy as np
from pytriton.model_config import Tensor

inputs = [
    Tensor(dtype=np.float32, shape=(-1,)),
]
output = [
    Tensor(dtype=np.float32, shape=(-1,)),
    Tensor(dtype=np.int32, shape=(-1,)),
]
```


The provided configuration creates the following tensors:

- Single input:
  - name: INPUT_1, data type: FLOAT32, shape=(-1,)
- Two outputs:
  - name: OUTPUT_1, data type: FLOAT32, shape=(-1,)
  - name: OUTPUT_2, data type: INT32, shape=(-1,)

The `-1` means a dynamic shape of the input or output.

To define the name of the input and its exact shape, the following definition can be used:

```python
import numpy as np
from pytriton.model_config import Tensor

inputs = [
    Tensor(name="image", dtype=np.float32, shape=(224, 224, 3)),
]
outputs = [
    Tensor(name="class", dtype=np.int32, shape=(1000,)),
]
```

This definition describes that the model has:

- a single input named `image` of size 224x224x3 and 32-bit floating-point data type
- a single output named `class` of size 1000 and 32-bit integer data type.

The `dtype` parameter can be either `numpy.dtype`, `numpy.dtype.type`, or `str`. For example:

```python
import numpy as np
from pytriton.model_config import Tensor

tensor1 = Tensor(name="tensor1", shape=(-1,), dtype=np.float32),
tensor2 = Tensor(name="tensor2", shape=(-1,), dtype=np.float32().dtype),
tensor3 = Tensor(name="tensor3", shape=(-1,), dtype="float32"),
```

!!! warning "dtype for bytes and string inputs/outputs"

    When using the `bytes` dtype, NumPy removes trailing `\x00` bytes.
    Therefore, for arbitrary bytes, it is required to use `object` dtype.

        > np.array([b"\xff\x00"])
        array([b'\xff'], dtype='|S2')

        > np.array([b"\xff\x00"], dtype=object)
        array([b'\xff\x00'], dtype=object)

    For ease of use, for encoded string values, users might use `bytes` dtype.

### Unrecoverable errors

When the model gets into a state where further inference is impossible,
you can throw [PyTritonUnrecoverableError][pytriton.exceptions.PyTritonUnrecoverableError]
from the inference callable. This will cause NVIDIA Triton Inference Server to shut down.
This might be useful when the model is deployed on a cluster in a multi-node setup. In that case
to recover the model you need to restart all "workers" on the cluster.

When the model gets into a state where further inference is impossible,
you can throw the [PyTritonUnrecoverableError][pytriton.exceptions.PyTritonUnrecoverableError]
from the inference callable. This will cause the NVIDIA Triton Inference Server to shut down.
This might be useful when the model is deployed on a cluster in a multi-node setup. In that case,
to recover the model, you need to restart all "workers" on the cluster.

```python
from typing import Dict
import numpy as np
from pytriton.decorators import batch
from pytriton.exceptions import PyTritonUnrecoverableError


@batch
def infer_fn(**inputs: np.ndarray) -> Dict[str, np.ndarray]:
    ...

    try:
        outputs = model(**inputs)
    except Exception as e:
        raise PyTritonUnrecoverableError(
            "Some unrecoverable error occurred, "
            "thus no further inferences are possible."
        ) from e

    ...
    return outputs
```

## Model Configuration

The additional model configuration for running a model through the Triton Inference Server can be provided in the `config`
argument in the `bind` method. This section describes the possible configuration enhancements.
The configuration of the model can be adjusted by overriding the defaults for the `ModelConfig` object.

```python
from pytriton.model_config.common import DynamicBatcher

class ModelConfig:
    batching: bool = True
    max_batch_size: int = 4
    batcher: DynamicBatcher = DynamicBatcher()
    response_cache: bool = False
```

### Batching

The batching feature collects one or more samples and passes them to the model together. The model processes
multiple samples at the same time and returns the output for all the samples processed together.

Batching can significantly improve throughput. Processing multiple samples at the same time leverages the benefits of
utilizing GPU performance for inference.

The Triton Inference Server is responsible for collecting multiple incoming requests into a single batch. The batch is
passed to the model, which improves the inference performance (throughput and latency). This feature is called
`dynamic batching`, which collects samples from multiple clients into a single batch processed by the model.

On the PyTriton side, the `infer_fn` obtain the fully created batch by Triton Inference Server so the only
responsibility is to perform computation and return the output.

By default, batching is enabled for the model. The default behavior for Triton is to have dynamic batching enabled.
If your model does not support batching, use `batching=False` to disable it in Triton.

### Maximal batch size

The maximal batch size defines the number of samples that can be processed at the same time by the model. This configuration
has an impact not only on throughput but also on memory usage, as a bigger batch means more data loaded to the memory
at the same time.

The `max_batch_size` has to be a value greater than or equal to 1.

### Dynamic batching

The dynamic batching is a Triton Inference Server feature and can be configured by defining the `DynamicBatcher`
object:

```python
from typing import Dict, Optional
from pytriton.model_config.common import QueuePolicy

class DynamicBatcher:
    max_queue_delay_microseconds: int = 0
    preferred_batch_size: Optional[list] = None
    preserve_ordering: bool = False
    priority_levels: int = 0
    default_priority_level: int = 0
    default_queue_policy: Optional[QueuePolicy] = None
    priority_queue_policy: Optional[Dict[int, QueuePolicy]] = None
```

More about dynamic batching can be found in
the [Triton Inference Server documentation](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher)
and [API spec](api.md)

### Response cache

The Triton Inference Server provides functionality to use a cached response for the model. To use the response cache:

- provide the `response_cache_byte_size` in `TritonConfig`
- set `response_cache=True` in `ModelConfig`

Example:

```python
import numpy as np

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

triton_config = TritonConfig(
    response_cache_byte_size=1024 * 1024,  # 1 MB
)

@batch
def _add_sub(**inputs):
    a_batch, b_batch = inputs.values()
    add_batch = a_batch + b_batch
    sub_batch = a_batch - b_batch
    return {"add": add_batch, "sub": sub_batch}

with Triton(config=triton_config) as triton:
    triton.bind(
        model_name="AddSub",
        infer_func=_add_sub,
        inputs=[Tensor(shape=(1,), dtype=np.float32), Tensor(shape=(1,), dtype=np.float32)],
        outputs=[Tensor(shape=(1,), dtype=np.float32), Tensor(shape=(1,), dtype=np.float32)],
        config=ModelConfig(max_batch_size=8, response_cache=True)
    )
    ...
```

## Deploying in Cluster

The library can be used inside containers and deployed on Kubernetes clusters. There are certain prerequisites and
information that would help deploy the library in your cluster.

### Health checks

The library uses the Triton Inference Server to handle HTTP/gRPC requests. Triton Server provides endpoints to validate if
the server is ready and in a healthy state. The following API endpoints can be used in your orchestrator to
control the application ready and live states:

- Ready: `/v2/health/ready`
- Live: `/v2/health/live`

### Exposing ports

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

### Configuring shared memory

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

### Specify container init process

You can use the [`--init` flag](https://docs.docker.com/engine/reference/run/#specify-an-init-process) of the `docker run`
command to indicate that an init process should be used as the PID 1 in the container.
Specifying an init process ensures that reaping zombie processes are performed inside the container. The reaping zombie
processes functionality is important in case of an unexpected error occurrence in scripts hosting PyTriton.