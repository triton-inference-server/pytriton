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

# Initialization

The following page provides more details about possible options for configuring the
[Triton Inference Server](https://github.com/triton-inference-server/server) and working with
block and non-blocking mode for tests and deployment.

## Configuring Triton

Connecting Python models with Triton Inference Server working in the current environment requires creating
a [Triton][pytriton.triton.Triton] object. This can be done by creating a context:

<!--pytest.mark.skip-->
```python
from pytriton.triton import Triton

with Triton() as triton:
    ...
```

or simply creating an object:

<!--pytest-codeblocks:cont-->
```python
from pytriton.triton import Triton

triton = Triton()
```

The Triton Inference Server behavior can be configured by passing [config][pytriton.triton.TritonConfig] parameter:

<!--pytest.mark.skip-->
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

## Triton Lifecycle Policy

Triton class has additional `lifecycle_policy` parameter of type `TritonLifecyclePolicy`. It indicates how the Triton Server should be managed.
It stores two flags: `TritonLifecyclePolicy.launch_triton_on_startup` and `TritonLifecyclePolicy.local_model_store`.
First one indicates if the Triton Server should be started on Triton object creation and the second one
indicates if the model store should be created in the local filesystem and polled by Triton or configuration should be passed to the Triton server and managed by it.

<!--pytest.mark.skip-->
```python
from pytriton.triton import Triton, TritonLifecyclePolicy

lifecycle_policy = TritonLifecyclePolicy(launch_triton_on_startup=False, local_model_store=True)
with Triton(triton_lifecycle_policy=lifecycle_policy) as triton:
    ...
    triton.serve()
```

Default values for `TritonLifecyclePolicy` flags are `launch_triton_on_startup=True` and `local_model_store=False`.
In this case Triton Server will be started on the Triton class instantiation (so user can bind models to running server interactively)
and the model store will be created in Triton server's filesystem and managed by it.

In some usage scenarios, it is necessary to prepare the model first in local model store and then start the Triton server (e.g. VertexAI flow).
In this case we use VerexAI's `TritonLifecyclePolicy` to manage the Triton server lifecycle.

`VertextAILifecyclePolicy = TritonLifecyclePolicy(launch_triton_on_startup=False, local_model_store=True)`

For easy of use, it is automatically set when `TritonConfig` is created with `allow_vertex_ai` parameter set to `True`.

`config = TritonConfig(allow_http=True, allow_vertex_ai=True, vertex_ai_port=8080)`

For details on how to use `TritonLifecyclePolicy` with VertexAI, see example [examples/add_sub_vertex_ai](../examples/add_sub_vertex_ai).


## Blocking mode

The blocking mode will stop the execution of the current thread and wait for incoming HTTP/gRPC requests for inference
execution. This mode makes your application behave as a pure server. The example of using blocking mode:

<!--pytest.mark.skip-->
```python
from pytriton.triton import Triton

with Triton() as triton:
    ...  # Load models here
    triton.serve()
```

## Background mode

The background mode runs Triton as a subprocess and does not block the execution of the current thread. In this mode, you can run
Triton Inference Server and interact with it from the current context. The example of using background mode:

```python
from pytriton.triton import Triton

triton = Triton()
...  # Load models here
triton.run()  # Triton Server started
print("This print will appear")
triton.stop()  # Triton Server stopped
```

## Filesystem usage

PyTriton needs to access the filesystem for two purposes:

  - to communicate with the Triton backend using file sockets,
  - storing copy of Triton backend and its binary dependencies.

PyTriton creates temporary folders called Workspaces, where it stores the file descriptors for these operations. By default, these folders are located in `$HOME/.cache/pytriton` directory. However, you can change this location by setting the `PYTRITON_HOME` environment variable.




