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


# Triton clients

The prerequisite for this page is to install PyTriton. You also need ```Linear``` model described in quick_start. You should run it so client can connect to it.

The clients section presents how to send requests to the Triton Inference Server using the PyTriton library.

## ModelClient

ModelClient is a simple client that can perform inference requests synchronously. You can use ModelClient to communicate with the deployed model using HTTP or gRPC protocol. You can specify the protocol when creating the ModelClient object.

For example, you can use ModelClient to send requests to a PyTorch model that performs linear regression:

<!--pytest.mark.skip-->
```python
import torch
from pytriton.client import ModelClient

# Create some input data as a numpy array
input1_data = torch.randn(128, 2).cpu().detach().numpy()

# Create a ModelClient object with the server address and model name
with ModelClient("localhost:8000", "Linear") as client:
    # Call the infer_batch method with the input data
    result_dict = client.infer_batch(input1_data)

# Print the result dictionary
print(result_dict)
```

You can also use ModelClient to send requests to a model that performs image classification. The example assumes that a model takes in an image and returns the top 5 predicted classes. This model is not included in the PyTriton library.

You need to convert the image to a numpy array and resize it to the expected input shape. You can use Pillow package to do this.

<!--pytest.mark.skip-->
```python
import numpy as np
from PIL import Image
from pytriton.client import ModelClient

# Create some input data as a numpy array of an image
img = Image.open("cat.jpg")
img = img.resize((224, 224))
input_data = np.array(img)

# Create a ModelClient object with the server address and model name
with ModelClient("localhost:8000", "ImageNet") as client:
    # Call the infer_batch method with the input data
    result_dict = client.infer_sample(input_data)

# Print the result dictionary
print(result_dict)
```

You need to install Pillow package to run the above example:
```bash
pip install Pillow
```

## FuturesModelClient

FuturesModelClient is a concurrent.futures based client that can perform inference requests in a parallel way. You can use FuturesModelClient to communicate with the deployed model using HTTP or gRPC protocol. You can specify the protocol when creating the FuturesModelClient object.

For example, you can use FuturesModelClient to send multiple requests to a text generation model that takes in text prompts and returns generated texts. The TextGen model is not included in the PyTriton library. The example assumes that the model returns a single output tensor with the generated text. The example also assumes that the model takes in a list of text prompts and returns a list of generated texts.

You need to convert the text prompts to numpy arrays of bytes using a tokenizer from transformers. You also need to detokenize the output texts using the same tokenizer:

<!--pytest.mark.skip-->
```python
import numpy as np
from pytriton.client import FuturesModelClient
from transformers import AutoTokenizer

# Create some input data as a list of text prompts
input_data_list_text = ["Write a haiku about winter.", "Summarize the article below in one sentence.", "Generate a catchy slogan for PyTriton."]

# Create a tokenizer from transformers
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Convert the text prompts to numpy arrays of bytes using the tokenizer
input_data_list = [np.array(tokenizer.encode(prompt)) for prompt in input_data_list_text]

# Create a FuturesModelClient object with the server address and model name
with FuturesModelClient("localhost:8000", "TextGen") as client:
    # Call the infer_sample method for each input data in the list and store the returned futures
    output_data_futures = [client.infer_sample(input_data) for input_data in input_data_list]
    # Wait for all the futures to complete and get the results
    output_data_list = [output_data_future.result() for output_data_future in output_data_futures]

# Print the list of result dictionaries
print(output_data_list)

# Detokenize the output texts using the tokenizer and print them
output_texts = [tokenizer.decode(output_data["OUTPUT_1"]) for output_data in output_data_list]
for output_text in output_texts:
    print(output_text)
```

You need to install transformers package to run the above example:
```bash
pip install transformers
```

You can also use FuturesModelClient to send multiple requests to an image classification model that takes in image data and returns class labels or probabilities. The ImageNet model is described above.

In this case, you can use the infer_batch method to send a batch of images as input and get a batch of outputs. You need to stack the images along the first dimension to form a batch. You can also print the class names corresponding to the output labels:

<!--pytest.mark.skip-->
``` python
import numpy as np
from PIL import Image
from pytriton.client import FuturesModelClient

# Create some input data as a list of lists of image arrays
input_data_list = []
for batch in [["cat.jpg", "dog.jpg", "bird.jpg"], ["car.jpg", "bike.jpg", "bus.jpg"], ["apple.jpg", "banana.jpg", "orange.jpg"]]:
  batch_data = []
  for filename in batch:
    img = Image.open(filename)
    img = img.resize((224, 224))
    img = np.array(img)
    batch_data.append(img)
  # Stack the images along the first dimension to form a batch
  batch_data = np.stack(batch_data, axis=0)
  input_data_list.append(batch_data)

# Create a list of class names for ImageNet
class_names = ["tench", "goldfish", "great white shark", ...]

# Create a FuturesModelClient object with the server address and model name
with FuturesModelClient("localhost:8000", "ImageNet") as client:
    # Call the infer_batch method for each input data in the list and store the returned futures
    output_data_futures = [client.infer_batch(input_data) for input_data in input_data_list]
    # Wait for all the futures to complete and get the results
    output_data_list = [output_data_future.result() for output_data_future in output_data_futures]

# Print the list of result dictionaries
print(output_data_list)

# Print the class names corresponding to the output labels for each batch
for output_data in output_data_list:
  output_labels = output_data["OUTPUT_1"]
  for output_label in output_labels:
    class_name = class_names[output_label]
    print(f"The image is classified as {class_name}.")
```

## Client timeouts

When creating a [ModelClient][pytriton.client.client.ModelClient] or [FuturesModelClient][pytriton.client.client.FuturesModelClient] object, you can specify the timeout for waiting until the server and model are ready using the `init_timeout_s` parameter. By default, the timeout is set to 5 minutes (300 seconds).

Example usage:

<!--pytest.mark.skip-->
```python
import numpy as np
from pytriton.client import ModelClient, FuturesModelClient

input1_data = np.random.randn(128, 2)
with ModelClient("localhost", "MyModel", init_timeout_s=120) as client:
    # Raises PyTritonClientTimeoutError if the server or model is not ready within the specified timeout
    result_dict = client.infer_batch(input1_data)


with FuturesModelClient("localhost", "MyModel", init_timeout_s=120) as client:
    future = client.infer_batch(input1_data)
    ...
    # It will raise `PyTritonClientTimeoutError` if the server is not ready and the model is not loaded within 120 seconds
    # from the time `infer_batch` was called by a thread from `ThreadPoolExecutor`
    result_dict = future.result()
```

You can disable the default behavior of waiting for the server and model to be ready during first inference request by setting `lazy_init` to `False`:

<!--pytest.mark.skip-->
```python
import numpy as np
from pytriton.client import ModelClient, FuturesModelClient

input1_data = np.random.randn(128, 2)

# will raise PyTritonClientTimeoutError if server is not ready and model loaded
# within 120 seconds during intialization of client
with ModelClient("localhost", "MyModel", init_timeout_s=120, lazy_init=False) as client:
    result_dict = client.infer_batch(input1_data)
```

You can specify the timeout for the client to wait for the inference response from the server.
The default timeout is 60 seconds. You can specify the timeout when creating the [ModelClient][pytriton.client.client.ModelClient] or [FuturesModelClient][pytriton.client.client.FuturesModelClient] object:

<!--pytest.mark.skip-->
```python
import numpy as np
from pytriton.client import ModelClient, FuturesModelClient

input1_data = np.random.randn(128, 2)
with ModelClient("localhost", "MyModel", inference_timeout_s=240) as client:
    # Raises `PyTritonClientTimeoutError` if the server does not respond to inference request within 240 seconds
    result_dict = client.infer_batch(input1_data)


with FuturesModelClient("localhost", "MyModel", inference_timeout_s=240) as client:
    future = client.infer_batch(input1_data)
    ...
    # Raises `PyTritonClientTimeoutError` if the server does not respond within 240 seconds
    # from the time `infer_batch` was called by a thread from `ThreadPoolExecutor`
    result_dict = future.result()
```

!!! warning "gRPC client timeout not fully supported"

    There are some missing features in the gRPC client that prevent it from working correctly with timeouts
    used during the wait for the server and model to be ready. This may cause the client to hang if the server
    doesn't respond with the current server or model state.

!!! info "Server side timeout not implemented"

    Currently, there is no support for server-side timeout. The server will continue to process the request even if the client timeout is reached.