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

The following section provide more details about possible options of configuring the
[Triton Inference Server](https://github.com/triton-inference-server/server),
configuring model for loading in Triton and deploying the solution in Docker container or clusters.

## Examples

Before you move to more advanced topics you may want to review examples which provide an implementation of various
models implementation (in JAX, Python, PyTorch and TensorFlow) deployed using the library.

You can also find the usage
of [Perf Analyzer](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md)
for profiling models (throughput, latency) once deployed using the solution.

For more, please review the [examples](examples.md) section.

## Configuring Triton

The [Triton][pytriton.triton.Triton] class is the base entrypoint for working with Triton Inference Server.

### Initialization

The connecting the Python models with Triton Inference Server working in the current environment requires creating
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

and through environment variables for example set as in the command below:

<!--pytest.mark.skip-->

```sh
PYTRITON_TRITON_CONFIG_LOG_VERBOSITY=4 python my_script.py
```

The order of precedence of configuration methods is:

- config defined through `config` parameter of [Triton][pytriton.triton.Triton] class `__init__` method.
- config defined in environment variables
- default [TritonConfig][pytriton.triton.TritonConfig] values

### Blocking mode

The blocking mode will stop the execution of current thread and wait for incoming HTTP/gRPC request for inference
execution. This mode make you application behave as a pure server. The example of using blocking mode:

<!--pytest.mark.skip-->

```python
from pytriton.triton import Triton

with Triton() as triton:
    ...  # Load models here
    triton.serve()
```

### Background mode

The background mode run Triton as subprocess and does not block execution of current thread. In this mode you can run
Triton Inference Server and interact with it from current context. The example of using background mode:

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

The Triton class provide method to load models one or multiple models to Triton server:

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

The `bind` method mandatory arguments:

- `model_name`: define under which name the model is available in Triton Inference Server
- `infer_func`: the lambda or function which obtain the data passed in request and return the output
- `inputs`: define the number, types and shapes for model inputs
- `outputs`: define the number, types and shapes for model outputs
- `config`: more customization for model deployment and behavior on Triton server

Once the `bind` method is called, the model is created in Triton Inference Server model store under
provided `model_name`.

### Inference Callable

The inference callable is an entrypoint for inference. This can be any callable that receive the data for
model inputs in form list of request dictionaries where input names are mapped into ndarrays.
Input can be also adapted to different more convenient form using set of decorators.
**More details about designing inference callable and using of decorators can be found
in separate section - [Inference Callable Design](inference_callable.md)**

In simplest implementation for functionality that pass input data on output the lambda can be used:

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

Multi-instance model inference is a mechanism for loading multiple instances of the same model and call
them alternately (to hide transfer overhead).

With `Triton` class it can be realised by providing the list of multiple inference callables to `Triton.bind`
in `infer_func` parameter.

The example present multiple instances of Linear PyTorch model loaded on separate device.

First, define the wrapper class for inference handler. The class initialization receive a model and device as the
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

Once the multiple callable objects are passed to `infer_func` the Triton server get information that multiple instances
of the same model has been created. The incoming requests are distributed among created instances. In our case executing
a `Linear` model instances loaded on CPU and GPU device.

### Inputs and Outputs

The integration of Python model requires to provide the inputs and outputs types of the model. This is required to
correctly map the input and output data passed through Triton Inference Server.

The simplest definition on model inputs and outputs expect to provide the type of data and the shape per input:

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

The provided configuration creates following tensors:

- Single input:
    - name: INPUT_1, data type: FLOAT32, shape = (-1)
- Two outputs:
    - name: OUTPUT_1, data type: FLOAT32, shape = (-1)
    - name: OUTPUT_2, data type: INT32, shape = (-1)

The `-1` mean a dynamic shape of the input or output.

In order to define the name of input and exact shapes the following definition can be used:

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

This definition describe that model has:

- single input of name `image` and size 224x224x3 and 32bit floating-point data type
- single output with name `class` of size 1000 and 32 integer data type.

The `dtype` parameter can be either `numpy.dtype`, `numpy.dtype.type` or `str`. Example:

```python
import numpy as np
from pytriton.model_config import Tensor

tensor1 = Tensor(name="tensor1", shape=(-1,), dtype=np.float32),
tensor2 = Tensor(name="tensor2", shape=(-1,), dtype=np.float32().dtype),
tensor3 = Tensor(name="tensor3", shape=(-1,), dtype="float32"),
```

!!! warning "dtype for bytes and string inputs/outputs"

    numpy removes trailing `\x00` bytes if `bytes` dtype is used,
    thus for arbitrary bytes it is required to use `object` dtype.

        > np.array([b"\xff\x00"])
        array([b'\xff'], dtype='|S2')

        > np.array([b"\xff\x00"], dtype=object)
        array([b'\xff\x00'], dtype=object)

    For ease of use, for encoded string values user might use `bytes` dtype.

### Unrecoverable errors

When the model gets into a state where further inference is impossible,
you can throw [PyTritonUnrecoverableError][pytriton.exceptions.PyTritonUnrecoverableError]
from the inference callable. This will cause NVIDIA Triton Inference Server to shut down.
This might be useful when the model is deployed on a cluster in a multi-node setup. In that case
to recover the model you need to restart all "workers" on the cluster.

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
            "Some unrecoverable error occurred thus no further inferences possible.") from e

    ...
    return outputs
```

## Model Configuration

The additional model configuration for running model through Triton Inference Server can be provided in `config`
argument in `bind` method. This section describe the possible configurations enhancements.
The config of model can be adjusted through overriding the defaults for `ModelConfig` object.

```python
from pytriton.model_config.common import DynamicBatcher

class ModelConfig:
    batching: bool = True
    max_batch_size: int = 4
    batcher: DynamicBatcher = DynamicBatcher()
    response_cache: bool = False
```

### Batching

The batching feature collect one or more samples and pass it to the model together. The model process
multiple samples at the same time and return the output for all the samples process together.

Batching can improve the throughput significantly. Processing multiple samples at the same time leverage benefits of
utilizing GPU performance for inference.

The Triton Inference Server is responsible to collect multiple incoming requests into a single batch. The batch is
passed to the model what improve the inference performance (throughput and latency). This feature called
`dynamic batching`- collect samples from multiple clients into a single batch processed by model.

On the PyTriton side, the `infer_fn` obtain the fully created batch by Triton Inference Server so the only
responsibility is to perform computation and return the output.

By default, batching is enabled for the model. The default behavior for Triton is to have dynamic batching enabled.
If your model not support batching use `batching=False` to disable it in Triton.

### Maximal batch size

The maximal batch size define the number of samples that can be processed at the same time by model. This configuration
has impact not only on throughput, but also at memory usage as bigger batch means more data loaded to the memory
at the same time.

The `max_batch_size` has to be a value greater or equal to 1.

### Dynamic batching

The dynamic batching is a Triton Inference Server feature and can be configured through defining `DynamicBatcher`
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

The Triton Inference Server provide functionality to use a cached response for the model. To use response cache:

- provide the `response_cache_byte_size` in `TritonConfig`
- set the `response_cache=True` in `ModelConfig`

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

The solution can be used inside containers and deployed on Kubernetes cluster. There are certain prerequisites and
information that would help deploy solution in your cluster.

### Health checks

The solution use Triton Inference Server to handle HTTP/gRPC requests. Triton Server provide endpoints to validate if
the server is ready and in healthy state. The following API endpoint can be used in your orchestrator solution to
control the application ready and live states:

- Ready: `/v2/health/ready`
- Live: `/v2/health/live`

### Exposing ports

The solution use Triton Inference Server that expose the HTTP, gRPC and metrics ports for communication. In default
configuration the following ports has to be exposed:

- 8000 for HTTP
- 8001 for gRPC
- 8002 for metrics

If the library inside Docker container the ports can be exposed through passing extra argument to `docker run`
command. Example of passing ports configuration:

<!--pytest.mark.skip-->

```shell
docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 {image}
```

In order to deploy container in Kubernetes, add ports definition for container in YAML deployment configuration:

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

The connection between Python callbacks and Triton Inference Server is using the shared memory passing data between the
processes. The Docker container the default amount of shared memory is 64MB which can be not enough to pass input and
output data of model. In order to increase available shared memory size pass additional flag to `docker run` command.
Example of increasing shared memory size to 8GB:

<!--pytest.mark.skip-->

```shell
docker run --shm-size 8GB {image}
```

In order to increase the shared memory size for Kubernetes the following configuration can be used:

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

You can use the [`--init` flag](https://docs.docker.com/engine/reference/run/#specify-an-init-process) of `docker run`
command to indicate that an init process should be used as the PID 1 in the container.
Specifying an init process ensures the reaping zombie processes are performed inside the container. The reaping zombie
processes functionality is important in case of an unexpected errors occurrence in scripts hosting PyTriton.