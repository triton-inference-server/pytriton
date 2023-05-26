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

# Example with downloaded input data

In the following example, we will demonstrate how to effectively utilize PyTriton with downloaded input data.
While the model itself does not possess any inputs, it utilize custom parameters or headers to extract a URL and download data from an external source, such as an S3 bucket.

The corresponding function can leverage the batch decorator since it does not rely on any parameters or headers.

## Example

<!--pytest.mark.skip-->
```python
import numpy as np
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

@batch
def model_infer_function(**inputs):
    ...

def request_infer_function(requests):
    for request in requests:
        image_url = request.parameters["custom_url"]
        image_jpeg = download(image_url)
        image_data = decompress(image_jpeg)
        request['images_data'] = image_data
    outputs = model_infer_function(requests)
    return outputs

with Triton(config=TritonConfig(http_header_forward_pattern="custom.*")) as triton:
    triton.bind(
        model_name="ImgModel",
        infer_func=request_infer_function,
        inputs=[],
        outputs=[Tensor(name="out", dtype=np.float32, shape=(-1,))],
        config=ModelConfig(max_batch_size=128),
    )
    triton.serve()
```

