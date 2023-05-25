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

<!--pytest.mark.skip-->

```python
from pytriton.triton import Triton

triton = Triton()
...  # Load models here
triton.run()  # Triton Server started
print("This print will appear")
triton.stop()  # Triton Server stopped
```
