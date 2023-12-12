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

# How to use PyTriton client to split a large input into smaller batches and send them to the server in parallel

In this article, you will learn how to use PyTriton clients to create a chunking client that can handle inputs that are larger than the maximum batch size of your model.

First, you need to create a model that can process a batch of inputs and produce a batch of outputs. For simplicity, let's assume that your model can only handle two inputs at a time. We will call this model "Batch2" and run it on a local Triton server.

Next, you need to create a client that can send requests to your model. In this article, we will use the FuturesModelClient, which returns a Future object for each request. A Future object is a placeholder that can be used to get the result or check the status of the request later.

However, there is a problem with using the FuturesModelClient directly. If you try to send an input that is larger than the maximum batch size of your model, you will get an error. For example, the following code tries to send an input of size 4 to the "Batch2" model, which has a maximum batch size of 2:

<!-- This codeblock is skipped because it will raise an exception -->

<!--pytest.mark.skip-->

```python
import numpy as np
from pytriton.client import FuturesModelClient

with FuturesModelClient(f"localhost", "Batch2") as client:
    input_tensor = np.zeros((4, 1), dtype=np.int32)
    print(client.infer_batch(input_tensor).result())
```

This code will raise an exception like this:

```
PyTritonClientInferenceServerError: Error occurred during inference request. Message: [request id: 0] inference request batch-size must be <= 2 for 'Batch2'
```

To solve this problem, we can use a ChunkingClient class that inherits from FuturesModelClient and overrides the infer_batch method. The ChunkingClient class takes a chunking strategy as an argument, which is a function that takes the input dictionary and the maximum batch size as parameters and yields smaller dictionaries of inputs. The default chunking strategy simply splits the input along the first dimension according to the maximum batch size. For example, if the input is `{"INPUT_1": np.zeros((5, 1), dtype=np.int32)}` and the maximum batch size is 2, then the default chunking strategy will yield:

```
{"INPUT_1": np.zeros((2, 1), dtype=np.int32)}
{"INPUT_1": np.zeros((2, 1), dtype=np.int32)}
{"INPUT_1": np.zeros((1, 1), dtype=np.int32)}
```

You can also define your own chunking strategy if you have more complex logic for splitting your input.


<!-- This readme is for testing code snippets with pytest. It has codeblocks marked with pytest-codeblocks:cont to combine them into one test. -->

<!-- First test -->
<!--
```python
# Import modules and define a batched inference function
import numpy as np
from pytriton.decorators import batch

@batch
def infer_fn(**inputs: np.ndarray):

    return [inputs["INPUT_1"]]

# Create a Triton server with the inference function and a model config
import numpy as np
from pytriton.triton import Triton, TritonConfig
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor

config = TritonConfig(log_verbose=0, log_file="/dev/null")

triton = Triton(config=config)
triton.bind(
    model_name="Batch2",
    infer_func=infer_fn,
    inputs=[
        Tensor(dtype=np.int32, shape=(1,)),  # sample containing single bytes value
    ],
    outputs=[
        Tensor(dtype=np.int32, shape=(1,)),
    ],
    config=ModelConfig(max_batch_size=2),
)

triton.run()
```
-->

<!--pytest-codeblocks:cont-->

```python
# Define a ChunkingClient class that inherits from FuturesModelClient and splits the input into smaller batches
import concurrent.futures
from pytriton.client import FuturesModelClient

class ChunkingClient(FuturesModelClient):
    def __init__(self, host, model_name, chunking_strategy=None, max_workers=None):
        super().__init__(host, model_name, max_workers=max_workers)
        self.chunking_strategy = chunking_strategy or self.default_chunking_strategy

    def default_chunking_strategy(self, kwargs, max_batch_size):
        # Split the input by the first dimension according to the max batch size
        size_of_dimention_0 = self.find_size_0(kwargs)
        for i in range(0, size_of_dimention_0, max_batch_size):
            yield {key: value[i:i+max_batch_size] for key, value in kwargs.items()}

    def find_size_0(self, kwargs):
        # Check the size of the first dimension of each tensor and raise errors if they are not consistent or valid
        size_of_dimention_0 = None
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                if value.ndim > 0:
                    size = value.shape[0]
                    if size_of_dimention_0 is None or size_of_dimention_0 == size:
                        size_of_dimention_0 = size
                    else:
                        raise ValueError("The tensors have different sizes at the first dimension")
                else:
                    raise ValueError("The tensor has no first dimension")
            else:
                raise TypeError("The value is not a numpy tensor")
        return size_of_dimention_0

    def infer_batch(self, *args, **kwargs):
        max_batch_size = self.model_config().result().max_batch_size
        # Send the smaller batches to the server in parallel and yield the futures with results
        futures = [super(ChunkingClient, self).infer_batch(*args, **chunk) for chunk in self.chunking_strategy(kwargs, max_batch_size)]
        for future in futures:
            yield future
```
To use the ChunkingClient class, you can create an instance of it and use it in a context manager. For example:

<!--pytest-codeblocks:cont-->

```python
# Use the ChunkingClient class with the default strategy to send an input of size 5 to the "Batch2" model
import numpy as np
from pytriton.client import FuturesModelClient
chunker_client = ChunkingClient("localhost", "Batch2")
results = []
with chunker_client as client:
    input_tensor = np.zeros((5, 1), dtype=np.int32)
    # Print the results of each future without concatenating them
    for future in client.infer_batch(INPUT_1=input_tensor):
        results.append(future.result())
print(results)
```

<!--pytest-codeblocks:cont-->
<!--
```python
# Stop the Triton server to free up resources
triton.stop()
# End of the first test

# Check results
assert len(results) == 3
two_zeros = np.array([[0],[0]], dtype=np.int32)
single_zero = np.array([[0]], dtype=np.int32)
assert np.all(results[0]['OUTPUT_1'] == two_zeros)
assert np.all(results[1]['OUTPUT_1'] == two_zeros)
assert np.all(results[2]['OUTPUT_1'] == single_zero)

```
-->

This code will print:

<!--
Real output from code also contains ``Signal (2) received.``, which is printed during ``triton.stop()`` so snippet can be tested agains output in sample.
-->
```
{'OUTPUT_1': array([[0],
       [0]], dtype=int32)}
{'OUTPUT_1': array([[0],
       [0]], dtype=int32)}
{'OUTPUT_1': array([[0]], dtype=int32)}
```

You can see that the input is split into three batches of sizes 2, 2, and 1, and each batch is sent to the server in parallel. The results are returned as futures that can be accessed individually without concatenating them.