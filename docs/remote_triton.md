<!--
Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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
# PyTriton remote mode

Remote mode is a way to use the PyTriton with the Triton Inference Server running remotely (at this moment
it must be deployed on the same machine, but may be launched in a different container).

To bind the model in remote mode, it is required to use the `RemoteTriton` class instead of `Triton`.
Only difference of using `RemoteTriton` is that it requires the triton `url` argument in the constructor.

## Example of binding a model in remote mode

Example below assumes that the Triton Inference Server is running on the same machine (launched with PyTriton
in separate python script).

RemoteTriton binds remote model to existing Triton Inference Server.

<!--pytest.mark.skip-->
```python
import numpy as np

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import RemoteTriton, TritonConfig

triton_config = TritonConfig(
    cache_config=[f"local,size={1024 * 1024}"],  # 1MB
)

@batch
def _add_sub(**inputs):
    a_batch, b_batch = inputs.values()
    add_batch = a_batch + b_batch
    sub_batch = a_batch - b_batch
    return {"add": add_batch, "sub": sub_batch}

with RemoteTriton(url='localhost') as triton:
    triton.bind(
        model_name="AddSub",
        infer_func=_add_sub,
        inputs=[Tensor(shape=(1,), dtype=np.float32), Tensor(shape=(1,), dtype=np.float32)],
        outputs=[Tensor(shape=(1,), dtype=np.float32), Tensor(shape=(1,), dtype=np.float32)],
        config=ModelConfig(max_batch_size=8, response_cache=True)
    )
    triton.serve()
```


