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

# Inference Callable Design

This document provides guidelines for creating an inference callable for PyTriton, which serves as the entry point for handling inference requests.

### Inference Callable

The inference callable is an entry point for handling inference requests. The interface of the inference callable assumes it receives a list of requests as dictionaries, where each dictionary represents one request mapping model input names to NumPy ndarrays.

There are two common implementations for inference callables:

1. Functions:

```python
import numpy as np
from typing import Dict, List

def infer_fn(requests: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    ...
```

2. Class:

<!--pytest-codeblocks:cont-->

```python
import numpy as np
from typing import Dict, List

class InferCallable:

    def __call__(self, requests: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
       ...
```

To use the inference callable with PyTriton, it must be bound to a Triton server instance using the `bind` method:

<!--pytest-codeblocks:cont-->

```python
import numpy as np
from pytriton.triton import Triton
from pytriton.model_config import ModelConfig, Tensor

with Triton() as triton:
    triton.bind(
        model_name="MyInferenceFn",
        infer_func=infer_fn,
        inputs=[Tensor(shape=(1, ), dtype=np.float32)],
        outputs=[Tensor(shape=(1, ), dtype=np.float32)],
        config=ModelConfig(max_batch_size=8)
    )

    infer_callable = InferCallable()
    triton.bind(
        model_name="MyInferenceCallable",
        infer_func=infer_callable,
        inputs=[Tensor(shape=(1, ), dtype=np.float32)],
        outputs=[Tensor(shape=(1, ), dtype=np.float32)],
        config=ModelConfig(max_batch_size=8)
    )
```

For more information on serving the inference callable, refer to the [Loading models section](inference_callable.md#loading-models) on Deploying Models page.

### Batching decorator

In many cases, it is more convenient to receive input already batched in the form of a NumPy array instead of a list of separate requests. For such cases, we have prepared the `@batch` decorator that adapts generic input into a batched form. It passes kwargs to the inference function where each named input contains a NumPy array with a batch of requests received by the Triton server.

Below, we show the difference between decorated and undecorated functions bound with Triton:

```python
import numpy as np
from pytriton.decorators import batch

# Sample input data with 2 requests - each with 2 inputs
input_data = [
  {'in1': np.array([[1, 1]]), 'in2': np.array([[2, 2]])},
  {'in1': np.array([[1, 2]]), 'in2': np.array([[2, 3]])}
]

def undecorated_identity_fn(requests):
  print(requests)
  # As expected, requests = [
  #     {'in1': np.array([[1, 1]]), 'in2': np.array([[2, 2]])},
  #     {'in1': np.array([[1, 2]]), 'in2': np.array([[2, 3]])},
  # ]
  results = requests
  return results

@batch
def decorated_identity_fn(in1, in2):
  print(in1, in2)
  # in1 = np.array([[1, 1], [1, 2]])
  # in2 = np.array([[2, 2], [2, 3]])
  # Inputs are batched by `@batch` decorator and passed to the function as kwargs, so they can be automatically mapped
  # with in1, in2 function parameters
  # Of course, we could get these params explicitly with **kwargs like this:
  # def decorated_infer_fn(**kwargs):
  return {"out1": in1, "out2": in2}

undecorated_identity_fn(input_data)
decorated_identity_fn(input_data)
```

More examples using the `@batch` decorator with different frameworks are shown below.

Example implementation for TensorFlow model:

```python
import numpy as np
import tensorflow as tf

from pytriton.decorators import batch

@batch
def infer_tf_fn(**inputs: np.ndarray):
    (images_batch,) = inputs.values()
    images_batch_tensor = tf.convert_to_tensor(images_batch)
    output1_batch = model.predict(images_batch_tensor)
    return [output1_batch]
```

Example implementation for PyTorch model:

```python
import numpy as np
import torch

from pytriton.decorators import batch

@batch
def infer_pt_fn(**inputs: np.ndarray):
    (input1_batch,) = inputs.values()
    input1_batch_tensor = torch.from_numpy(input1_batch).to("cuda")
    output1_batch_tensor = model(input1_batch_tensor)
    output1_batch = output1_batch_tensor.cpu().detach().numpy()
    return [output1_batch]

```

Example implementation with [named inputs and outputs](deploying_models.md#inputs-and-outputs):

```python
import numpy as np
from pytriton.decorators import batch

@batch
def add_subtract_fn(a: np.ndarray, b: np.ndarray):
    return {"add": a + b, "sub": a - b}

@batch
def multiply_fn(**inputs: np.ndarray):
    a = inputs["a"]
    b = inputs["b"]
    return [a * b]
```

Example implementation with strings:

```python
import numpy as np
from transformers import pipeline
from pytriton.decorators import batch

CLASSIFIER = pipeline("zero-shot-classification", model="facebook/bart-base", device=0)

@batch
def classify_text_fn(text_array: np.ndarray):
    text = text_array[0]  # text_array contains one string at index 0
    text = text.decode("utf-8")  # string is stored in byte array encoded in utf-8
    result = CLASSIFIER(text)
    return [np.array(result)]  # return statistics generated by classifier
```

### Additional decorators for adapting input to user needs

We have prepared several useful decorators for converting generic request input into common user needs. You can create custom decorators tailored to your requirements and chain them with other decorators.

Our standard decorators implement three types of interfaces:

a) Receive a list of request dictionaries and return a list of response dictionaries (dictionaries map input/output names to NumPy arrays)

- `@group_by_keys` - groups requests with the same set of keys and calls the wrapped function for each group separately. This decorator is convenient to use before batching because the batching decorator requires a consistent set of inputs as it stacks them into batches.

  ```python
  from pytriton.decorators import batch, group_by_keys

  @group_by_keys
  @batch
  def infer_fn(mandatory_input, optional_input=None):
      # perform inference
      pass
  ```

- `@group_by_values(*keys)` - groups requests with the same input value (for selected keys) and calls the wrapped function for each group separately. This decorator is particularly useful with models requiring dynamic parameters sent by users, such as temperature. In this case, we want to run the model only for requests with the same temperature value.

  ```python
  from pytriton.decorators import batch, group_by_values

  @batch
  @group_by_values('temperature')
  def infer_fn(mandatory_input, temperature):
      # perform inference
      pass
  ```

- `@fill_optionals(**defaults)` - fills missing inputs in requests with default values provided by the user. If model owners have default values for some optional parameters, it's a good idea to provide them at the beginning, so other decorators can create larger consistent groups and send them to the inference callable.

  ```python
  import numpy as np
  from pytriton.decorators import batch, fill_optionals, group_by_values

  @fill_optionals(temperature=np.array([10.0]))
  @batch
  @group_by_values('temperature')
  def infer_fn(mandatory_input, temperature):
      # perform inference
      pass
  ```

b) Receive a list of request dictionaries and return a dictionary that maps input names to arrays (passes the dictionary to the wrapped infer function as named arguments - `kwargs`):

- `@batch` - generates a batch from input requests.
- `@sample` - takes the first request and converts it into named inputs. This decorator is useful with non-batching models. Instead of a one-element list of requests, we get named inputs - `kwargs` (usage is shown in previous examples).

c) Receive a batch (a dictionary that maps input names to arrays) and return a batch after some processing:

- `@pad_batch` - appends the last row to the input multiple times to achieve the desired batch size (preferred batch size or max batch size from the model config, whichever is closer to the current input size).

  ```python
  from pytriton.decorators import batch, pad_batch

  @batch
  @pad_batch
  def infer_fn(mandatory_input):
      # this model requires mandatory_input batch to be the size provided in the model config
      pass
  ```

- `@first_value` - this decorator takes the first elements from batches for selected inputs specified by the `keys` parameter.
  If the value is a one-element array, it is converted to a scalar value.
  This decorator is convenient to use with dynamic model parameters that users send in requests.
  You can use `@group_by_values` before to have batches with the same values in each batch.

  ```python
  import numpy as np
  from pytriton.decorators import batch, fill_optionals, first_value, group_by_values

  @fill_optionals(temperature=np.array([10.0]))
  @batch
  @group_by_values('temperature')
  @first_value('temperature')
  def infer_fn(mandatory_input, temperature):
      # perform inference with scalar temperature=10
      pass
  ```

d) The `@triton_context` decorator provides an additional argument called `triton_context`,
   from which you can read the model config.

   ```python
   from pytriton.decorators import triton_context

   @triton_context
   def infer_fn(input_list, **kwargs):
       model_config = kwargs['triton_context'].model_config
       # perform inference using some information from model_config
       pass
  ```

Here is an example of stacking multiple decorators together.
We recommend starting with type a) decorators, followed by types b) and c).
Place the `@triton_context` decorator last in the chain.

```python
import numpy as np
from pytriton.decorators import batch, fill_optionals, first_value, group_by_keys, group_by_values, triton_context

@fill_optionals(temperature=np.array([10.0]))
@group_by_keys
@batch
@group_by_values('temperature')
@first_value('temperature')
@triton_context
def infer(triton_context, mandatory_input, temperature, opt1=None, opt2=None):
    model_config = triton_context.model_config
    # perform inference using:
    #   - some information from model_config
    #   - scalar temperature value
    #   - optional parameters opt1 and/or opt2
```
