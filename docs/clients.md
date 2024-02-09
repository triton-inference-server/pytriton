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

PyTriton client is a user-friendly tool designed to communicate with the Triton Inference Server effortlessly. It manages the technical details for you, allowing you to concentrate on your data and the outcomes you aim to achieve. Here's how it assists:

1. **Fetching Model Configuration:** The client retrieves details about the model from the server, such as the shape and names of the input and output data tensors. This step is crucial for preparing your data correctly and for interpreting the response. This functionality is encapsulated in the [ModelClient][pytriton.client.ModelClient] class.

2. **Sending Requests:** Utilizing the model information, the client generates an inference request by mapping arguments you pass to the [infer_sample][pytriton.client.ModelClient.infer_sample] or [infer_batch][pytriton.client.ModelClient.infer_batch] methods to model inputs. It sends your data to the Triton server, requesting the model to perform inference. Arguments can be passed as positional or keyword arguments (mixing them is not allowed), and the client handles the rest.

3. **Returning Responses:** It then delivers the model's response back to you. It decodes inputs as numpy arrays and maps model outputs to dictionary elements returned to you from the `infer_sample` or `infer_batch` methods. It also removes the batch dimension if it was added by the client.

This process might introduce a bit of delay due to the extra step of fetching model configuration. However, you can minimize this by reusing the PyTriton client for multiple requests or by setting it up with pre-loaded model configuration if you have it.

PyTriton includes five specialized high-level clients to cater to different needs:

- **[ModelClient][pytriton.client.ModelClient]:** A straightforward, synchronous client for simple request-response operations.
- **[FuturesModelClient][pytriton.client.FuturesModelClient]:** A multithreaded client that handles multiple requests in parallel, speeding up operations.
- **[DecoupledModelClient][pytriton.client.DecoupledModelClient]:** A synchronous client designed for decoupled models, which allow for flexible interaction patterns with the Triton server.
- **[AsyncioModelClient][pytriton.client.AsyncioModelClient]:** An asynchronous client that works well with Python's asyncio for efficient concurrent operations.
- **[AsyncioDecoupledModelClient][pytriton.client.AsyncioDecoupledModelClient]:** An asyncio-compatible client specifically for working with decoupled models asynchronously.

PyTriton clients used [tritonclient](https://github.com/triton-inference-server/client) package from Triton. It is a Python client library for Triton Inference Server. It provides low level API for communicating with the server using HTTP or gRPC protocol. PyTriton clients are built on top of tritonclient and provide high level API for communicating with the server. Not all features of tritonclient are available in PyTriton clients. If you need more control over the communication with the server, you can use tritonclient directly.


## ModelClient

ModelClient is a simple client that can perform inference requests synchronously. You can use ModelClient to communicate with the deployed model using HTTP or gRPC protocol. You can specify the protocol when creating the ModelClient object.

You need ```Linear``` model described in quick_start. You should run it so client can connect to it.

For example, you can use ModelClient to send requests to a PyTorch model that performs linear regression:

<!-- This readme is for testing code snippets with pytest. It has codeblocks marked with pytest-codeblocks:cont to combine them into one test. -->

<!-- First test -->
<!--
```python

import torch

model = torch.nn.Linear(2, 3).eval()

import numpy as np
from pytriton.decorators import batch


@batch
def infer_fn(**inputs: np.ndarray):
    (input1_batch,) = inputs.values()
    input1_batch_tensor = torch.from_numpy(input1_batch)
    output1_batch_tensor = model(input1_batch_tensor)  # Calling the Python model inference
    output1_batch = output1_batch_tensor.detach().numpy()
    return [output1_batch]


from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

# Connecting inference callable with Triton Inference Server
triton = Triton()
# Load model into Triton Inference Server
triton.bind(
    model_name="Linear",
    infer_func=infer_fn,
    inputs=[
        Tensor(dtype=np.float32, shape=(-1,)),
    ],
    outputs=[
        Tensor(dtype=np.float32, shape=(-1,)),
    ],
    config=ModelConfig(max_batch_size=128)
)


triton.run()
```
-->


<!--pytest-codeblocks:cont-->

```python
import torch
from pytriton.client import ModelClient

# Create some input data as a numpy array
input1_data = torch.randn(128, 2).cpu().detach().numpy()

# Create a ModelClient object with the server address and model name
client = ModelClient("localhost:8000", "Linear")
# Call the infer_batch method with the input data
result_dict = client.infer_batch(input1_data)
# Close the client to release the resources
client.close()

# Print the result dictionary
print(result_dict)
```

<!--pytest-codeblocks:cont-->
<!--
```python
assert result_dict["OUTPUT_1"].shape == (128, 3)
```
-->

URL `localhost:8000` is the default address for Triton server HTTP protocol. If you have a different address, you should replace it with the correct one. You can also use the gRPC protocol by putting `grpc` in address string:

<!--pytest-codeblocks:cont-->
```python
client = ModelClient("grpc://localhost", "Linear")
```

<!--pytest-codeblocks:cont-->
<!--
```python
# Check that client is valid
result_dict = client.infer_batch(input1_data)
assert result_dict["OUTPUT_1"].shape == (128, 3)

# Stop the Triton server to free up resources
triton.stop()
# End of the first test

```
-->



You can omit port number if it is default for HTTP or gRPC protocol. Default port for HTTP is `8000` and for gRPC is `8001`.


You can also use ModelClient to send requests to a model that performs image classification. The example assumes that a model takes in an image and returns the top 5 predicted classes. This model is not included in the PyTriton library.

You need to convert the image to a numpy array and resize it to the expected input shape. You can use Pillow package to do this.

You need to install Pillow package to run the example:
```bash
pip install Pillow
```


<!--
```python
import numpy as np
from PIL import Image

# Create a new images with the same size as the input data
test_img = Image.new("RGB", (224, 224))
# Get the pixels object of the image
pixels = test_img.load()
test_img.save("cat.jpg")

```
--->


<!--
```python

import numpy as np
from pytriton.decorators import batch


@batch
def infer_fn(image):
    assert image.shape == (1, 224, 224, 3), f"Expected image shape (224, 224, 3), got {image.shape}"
    # This can implement actual inference logic for image classification
    return [np.char.encode(["cat"], "utf-8")]


from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

# Connecting inference callable with Triton Inference Server
triton = Triton()
# Load model into Triton Inference Server
triton.bind(
    model_name="ImageNet",
    infer_func=infer_fn,
    inputs=[
        Tensor(name="image", dtype=np.uint8, shape=(224, 224, 3)),
    ],
    outputs=[
        Tensor(name="class", dtype=np.bytes_, shape=(-1,)),
    ],
    config=ModelConfig(max_batch_size=2)
)


triton.run()
```
-->

<!--pytest-codeblocks:cont-->
```python
import numpy as np
from PIL import Image
from pytriton.client import ModelClient

# Create some input data as a numpy array of an image
img = Image.open("cat.jpg")
img = img.resize((224, 224))
input_data = np.array(img)

# Create a ModelClient object with the server address and model name
client = ModelClient("localhost:8000", "ImageNet")
# Call the infer_sample method with the input data
result_dict = client.infer_sample(input_data)
# Close the client to release the resources
client.close()

# Print the result dictionary
print(result_dict)
```

<!--pytest-codeblocks:cont-->
<!--
```python
# This is cleanup code for codeblocks testing
triton.stop()

# Remove test cat image.
import os
os.remove("cat.jpg")

assert np.char.decode(result_dict["class"], "utf-8") == "cat"

```
-->




## FuturesModelClient

FuturesModelClient is a concurrent.futures based client that can perform inference requests in a parallel way. You can use FuturesModelClient to communicate with the deployed model using HTTP or gRPC protocol. You can specify the protocol when creating the FuturesModelClient object.

For example, you can use FuturesModelClient to send multiple requests to a text generation model that takes in text prompts and returns generated texts. The TextGen model is not included in the PyTriton library. The example assumes that the model returns a single output tensor with the generated text. The example also assumes that the model takes in a list of text prompts and returns a list of generated texts.

You need to convert the text prompts to numpy arrays of bytes using a tokenizer from transformers. You also need to detokenize the output texts using the same tokenizer:

You need to install torch and transformers package to run the example:
```bash
pip install torch transformers
```

<!-- This is code to setup GPT2 model in Triton to test readme in codeblocks -->
<!--
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
model = GPT2LMHeadModel.from_pretrained("gpt2")

import numpy as np
from pytriton.decorators import batch

@batch
def infer_fn(input_ids):
    output_ids = model.generate(torch.tensor(input_ids))
    return [output_ids.numpy()]


from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

# Connecting inference callable with Triton Inference Server
triton = Triton()
# Load model into Triton Inference Server
triton.bind(
    model_name="TextGen",
    infer_func=infer_fn,
    inputs=[
        Tensor(name="input_ids", dtype=np.int64, shape=(-1,)),
    ],
    outputs=[
        Tensor(dtype=np.int64, shape=(-1,)),
    ],
    config=ModelConfig(max_batch_size=16)
)

triton.run()

```
-->


<!--pytest-codeblocks:cont-->
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

# Print tokens
print(output_data_list)

# Detokenize the output texts using the tokenizer and print them
output_texts = [tokenizer.decode(output_data["OUTPUT_1"]) for output_data in output_data_list]
for output_text in output_texts:
    print(output_text)
```

<!--pytest-codeblocks:cont-->
<!--
```python
# This is cleanup code for codeblocks testing
triton.stop()
# End of the second test
```
-->


You can also use FuturesModelClient to send multiple requests to an image classification model that takes in image data and returns class labels or probabilities. The ImageNet model is described above.

In this case, you can use the infer_batch method to send a batch of images as input and get a batch of outputs. You need to stack the images along the first dimension to form a batch. You can also print the class names corresponding to the output labels:

<!--
```python
import numpy as np
from PIL import Image

# Define a list of colors as RGB tuples
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 128, 128), (255, 255, 255), (0, 0, 0)]
# Define a list of image names
images_names = [["cat.jpg", "dog.jpg", "bird.jpg"], ["car.jpg", "bike.jpg", "bus.jpg"], ["apple.jpg", "banana.jpg", "orange.jpg"]]
# Loop over the image names and colors
for batch, color in zip(images_names, colors):
    # Create a new image with the same size as the input data and the given color
    test_img = Image.new("RGB", (224, 224), color)
    for image in batch:
        test_img.save(image)


import numpy as np
from pytriton.decorators import batch


def average_color(image):
    image = np.array(image)
    # Compute the mean of each color channel
    r = image[:,:,0].mean()
    g = image[:,:,1].mean()
    b = image[:,:,2].mean()
    return (r, g, b)

# Define a function to classify an image based on its average color
def classify_image(image):
    avg_color = average_color(image)
    # Define the thresholds for each category
    tench_threshold = 150
    goldfish_threshold = 100
    shark_threshold = 50
    if avg_color[0] > tench_threshold:
        label = 0
    elif avg_color[1] > goldfish_threshold:
        label = 1
    elif avg_color[2] > shark_threshold:
        label = 2
    else:
        assert False, f"Unexpected average color {avg_color}"
    return label

@batch
def infer_fn(image):
    assert image.shape[1:] == (224, 224, 3), f"Expected image shape (224, 224, 3), got {image.shape}"
    labels = []
    for img in image:
        label = classify_image(img)
        labels.append(label)
    # Convert the list to a numpy array and return it
    return [np.array(labels)]


from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

# Connecting inference callable with Triton Inference Server
triton = Triton()
# Load model into Triton Inference Server
triton.bind(
    model_name="ImageNet",
    infer_func=infer_fn,
    inputs=[
        Tensor(name="image", dtype=np.uint8, shape=(224, 224, 3)),
    ],
    outputs=[
        Tensor(dtype=np.int32, shape=(-1,)),
    ],
    config=ModelConfig(max_batch_size=3)
)


triton.run()
```
-->

<!--pytest-codeblocks:cont-->
```python
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

<!--pytest-codeblocks:cont-->
<!--
```python
# This is cleanup code for codeblocks testing
triton.stop()

# Remove test cat image.
import os
for batch in images_names:
    for image in batch:
        os.remove(image)

expected_result = [["tench", "tench", "tench",],
                   ["goldfish", "goldfish", "goldfish"],
                   ["great white shark", "great white shark", "great white shark",],]

# Test outputs
for output, expected_output in zip(output_data_list, expected_result):
    for c_id, label in zip(output["OUTPUT_1"], expected_output):
        assert class_names[c_id] == label, f"Expected {label}, got {class_names[c_id]}"

```
-->


## AsyncioModelClient

AsyncioModelClient is an asynchronous client that can perform inference requests using the asyncio library. You can use AsyncioModelClient to communicate with the deployed model using HTTP or gRPC protocol. You can specify the protocol when creating the AsyncioModelClient object.

For example, you can use AsyncioModelClient to send requests to a PyTorch model that performs linear regression:

<!-- Setup Triton server to test client code below -->

<!--
```python

import torch

model = torch.nn.Linear(2, 3).eval()

import numpy as np
from pytriton.decorators import batch


@batch
def infer_fn(**inputs: np.ndarray):
    (input1_batch,) = inputs.values()
    input1_batch_tensor = torch.from_numpy(input1_batch)
    output1_batch_tensor = model(input1_batch_tensor)  # Calling the Python model inference
    output1_batch = output1_batch_tensor.detach().numpy()
    return [output1_batch]


from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

# Connecting inference callable with Triton Inference Server
triton = Triton()
# Load model into Triton Inference Server
triton.bind(
    model_name="Linear",
    infer_func=infer_fn,
    inputs=[
        Tensor(dtype=np.float32, shape=(-1,)),
    ],
    outputs=[
        Tensor(dtype=np.float32, shape=(-1,)),
    ],
    config=ModelConfig(max_batch_size=128)
)


triton.run()
```
-->


<!--pytest-codeblocks:cont-->
```python
import torch
from pytriton.client import AsyncioModelClient

# Make the code async by adding async before the function definition
async def main():
    # Create some input data as a numpy array
    input1_data = torch.randn(2).cpu().detach().numpy()

    # Create an AsyncioModelClient object with the server address and model name
    client = AsyncioModelClient("localhost:8000", "Linear")
    # Call the infer_sample method with the input data
    result_dict = await client.infer_sample(input1_data)
    # Close the client to release the resources
    await client.close()

    # Print the result dictionary
    print(result_dict)

# Run the code as a coroutine using asyncio.run()
import asyncio
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
```

<!--pytest-codeblocks:cont-->
<!--
```python
# Stop the Triton server to free up resources
triton.stop()
# End of asyncio test

```
-->


You can also use FastAPI to create a web application that exposes the results of inference at an HTTP endpoint. FastAPI is a modern, fast, web framework for building APIs with Python 3.6+ based on standard Python type hints.

To use FastAPI, you need to install it with:

```bash
pip install fastapi
```

You also need an ASGI server, for production such as Uvicorn or Hypercorn.

To install Uvicorn, run:

<!--pytest.mark.skip-->
```bash
pip install uvicorn[standard]
```

The `uvicorn` uses port `8000` as default for web server. Triton server default port is also `8000` for HTTP protocol. You can change uvicorn port by using `--port` option. PyTriton also supports custom ports configuration for Triton server. The class `TritonConfig` contains parameters for ports configuration. You can pass it to `Triton` during initialization:


<!-- Add imports omitted below -->
<!--
```python
from pytriton.triton import Triton, TritonConfig
```
-->


<!--pytest-codeblocks:cont-->
```python
config = TritonConfig(http_port=8015)
triton_server = Triton(config=config)
```

You can use this `triton_server` object to bind your inference model and run HTTP endpoint from Triton Inference Server at port `8015`.


Then you can create a FastAPI app that uses the AsyncioModelClient to perform inference and return the results as JSON:

<!-- FastAPI server is very hard to setup in codeblocks testing. So we skip it. -->

<!--pytest.mark.skip-->
```python
from fastapi import FastAPI
import torch
from pytriton.client import AsyncioModelClient

# Create an AsyncioModelClient object with the server address and model name
config_client = None

app = FastAPI()

@app.get("/predict")
async def predict():
    # Create some input data as a numpy array
    input1_data = torch.randn(2).cpu().detach().numpy()

    # Create an AsyncioModelClient object with the server address and model name and store it in a global variable
    global config_client
    if not config_client:
        config_client = AsyncioModelClient("localhost:8000", "Linear")
        await config_client.model_config

    # Create an AsyncioModelClient object from existing client to avoid pulling config from server
    async with AsyncioModelClient.from_existing_client(config_client) as request_client:
        # Call the infer_sample method with the input data
        result_dict = await request_client.infer_sample(input1_data)

    output_dict = {key: value.tolist() for key, value in result_dict.items()}

    # Return the result dictionary as JSON
    return output_dict


@app.on_event("shutdown")
async def shutdown():
    # Close the client to release the resources
    await config_client.close()
```

Save this file as `main.py`.

To run the app, use the command:

<!--pytest.mark.skip-->
```bash
uvicorn main:app --reload --port 8015
```

You can then access the endpoint at `http://127.0.0.1:8015/predict` and see the JSON response.

You can also check the interactive API documentation at `http://127.0.0.1:8015/docs`.

You can test your server using curl:

<!--pytest.mark.skip-->
```bash
curl -X 'GET' \
  'http://127.0.0.1:8015/predict' \
  -H 'accept: application/json'
```

Command will print three random numbers:
<!--pytest.mark.skip-->
```python
[-0.2608422636985779,-0.6435106992721558,-0.3492531180381775]
```

For more information about FastAPI and Uvicorn, check out these links:

- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [Uvicorn documentation](https://www.uvicorn.org/)

## Decoupled models and clients

You can use the PyTriton library to create a decoupled model and client. A decoupled model is a model that is decoupled from batching and other features of the Triton Inference Server. It can receive many requests in parallel and perform inference on each request independently. A decoupled model can send multiple responses to single requests. A decoupled client is a client that can receive multiple responses from a single request. See [document](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/decoupled_models.md) about decoupled models for more information.

ModelClient, AsyncioModelClient, and FuturesModelClient refuse to send requests to decoupled models and raise an exception. You can use DecoupledModelClient and AsyncioDecoupledModelClient to send requests to decoupled models. You can only communicate over gRPC protocol with decoupled models.

!!! info "The Generate Extension is provisional and likely to change in future versions"

You can use generate stream endpoint in Triton Inference Server to send requests to decoupled models. See [Generate Extension](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md).

For example, you can use DecoupledModelClient to send requests to GPT2 using stream endpoint:
<!--
```python
import concurrent.futures
import numpy as np
import transformers
import pytriton

model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
streamer = transformers.TextIteratorStreamer(tokenizer)
executor = concurrent.futures.ThreadPoolExecutor()

def infer_func(inputs):
    text = (np.char.decode(inputs[0]["inputs"].astype("bytes"),"utf-8").tolist())[0]
    generate_kwargs = dict(
        **tokenizer.batch_encode_plus(text, return_tensors="pt", padding=True),
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id
    )
    generate_future = executor.submit(model.generate, **generate_kwargs)
    # Your inference callable is generator that yields partial responses
    tokens = []
    for token in streamer:
        tokens.append(token)
        yield [{"response": np.array(["".join(tokens).encode("utf-8")])}]
    generate_future.result()

triton = pytriton.triton.Triton()
triton.bind(
    model_name="streaming_bot",
    infer_func=infer_func,
    inputs=[
        pytriton.model_config.Tensor(name="inputs", dtype=bytes, shape=(1,)),
    ],
    outputs=[
        pytriton.model_config.Tensor(name="response", dtype=bytes, shape=(1,)),
    ],
    config=pytriton.model_config.ModelConfig(decoupled=True), # Set decoupled=True to enable decoupled mode
)
triton.run()
```
-->

<!--pytest-codeblocks:cont-->
```python
import torch
from pytriton.client import DecoupledModelClient

client = DecoupledModelClient("grpc://localhost", "streaming_bot")
iterator = client.infer_sample(np.array(["AI answers to".encode('utf-8')]))
result = list(iterator)
print(result)
client.close()
```

Asynchronous version of DecoupledModelClient is AsyncioDecoupledModelClient. You can use it to send requests to decoupled models using asyncio library:

<!--pytest-codeblocks:cont-->
```python
from pytriton.client import AsyncioDecoupledModelClient
# Make the code async by adding async before the function definition
async def main():
    client = AsyncioDecoupledModelClient("grpc://localhost", "streaming_bot")
    async for answer in client.infer_sample(np.array(["AI answers to".encode('utf-8')])):
        print(answer)
    await client.close()

# Run the code as a coroutine using asyncio.run()
import asyncio
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
```

<!--pytest-codeblocks:cont-->
<!--
```python
triton.stop()

assert result[-1]["response"] == b'AI answers to the question, "What is the best way to get rid of the problem?"\n'
```
-->

This will print the following output:
```
{'response': b'AI answers '},
{'response': b'AI answers to '},
{'response': b'AI answers to the '},
{'response': b'AI answers to the '},
{'response': b'AI answers to the question, '},
{'response': b'AI answers to the question, '},
{'response': b'AI answers to the question, "What '},
{'response': b'AI answers to the question, "What is '},
{'response': b'AI answers to the question, "What is the '},
{'response': b'AI answers to the question, "What is the best '},
{'response': b'AI answers to the question, "What is the best way '},
{'response': b'AI answers to the question, "What is the best way to '},
{'response': b'AI answers to the question, "What is the best way to get '},
{'response': b'AI answers to the question, "What is the best way to get rid '},
{'response': b'AI answers to the question, "What is the best way to get rid of '},
{'response': b'AI answers to the question, "What is the best way to get rid of the '},
{'response': b'AI answers to the question, "What is the best way to get rid of the '},
{'response': b'AI answers to the question, "What is the best way to get rid of the problem?"\n'},
{'response': b'AI answers to the question, "What is the best way to get rid of the problem?"\n'}
```

Each response contains more tokens than the previous one. The last response contains the full generated text.





## Client timeouts

When creating a [ModelClient][pytriton.client.client.ModelClient] or [FuturesModelClient][pytriton.client.client.FuturesModelClient] object, you can specify the timeout for waiting until the server and model are ready using the `init_timeout_s` parameter. By default, the timeout is set to 5 minutes (300 seconds).

Example usage:

<!-- Timeout test -->
<!--
```python

import torch

model = torch.nn.Linear(2, 3).eval()

import numpy as np
from pytriton.decorators import batch


@batch
def infer_fn(**inputs: np.ndarray):
    (input1_batch,) = inputs.values()
    input1_batch_tensor = torch.from_numpy(input1_batch).float()
    output1_batch_tensor = model(input1_batch_tensor)  # Calling the Python model inference
    output1_batch = output1_batch_tensor.detach().double().numpy()
    return [output1_batch]


from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

# Connecting inference callable with Triton Inference Server
triton = Triton()
# Load model into Triton Inference Server
triton.bind(
    model_name="MyModel",
    infer_func=infer_fn,
    inputs=[
        Tensor(dtype=np.float64, shape=(-1,)),
    ],
    outputs=[
        Tensor(dtype=np.float64, shape=(-1,)),
    ],
    config=ModelConfig(max_batch_size=128)
)


triton.run()
```
-->

<!--pytest-codeblocks:cont-->

```python
import numpy as np
from pytriton.client import ModelClient, FuturesModelClient

input1_data = np.random.randn(128, 2)
client = ModelClient("localhost", "MyModel", init_timeout_s=120)
# Raises PyTritonClientTimeoutError if the server or model is not ready within the specified timeout
result_dict = client.infer_batch(input1_data)
client.close()


with FuturesModelClient("localhost", "MyModel", init_timeout_s=120) as client:
    future = client.infer_batch(input1_data)
    #...
    # It will raise `PyTritonClientTimeoutError` if the server is not ready and the model is not loaded within 120 seconds
    # from the time `infer_batch` was called by a thread from `ThreadPoolExecutor`
    result_dict = future.result()
```

You can disable the default behavior of waiting for the server and model to be ready during first inference request by setting `lazy_init` to `False`:

<!--pytest-codeblocks:cont-->
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

<!--pytest-codeblocks:cont-->
```python
import numpy as np
from pytriton.client import ModelClient, FuturesModelClient

input1_data = np.random.randn(128, 2)
client = ModelClient("localhost", "MyModel", inference_timeout_s=240)
# Raises `PyTritonClientTimeoutError` if the server does not respond to inference request within 240 seconds
result_dict = client.infer_batch(input1_data)
client.close()


with FuturesModelClient("localhost", "MyModel", inference_timeout_s=240) as client:
    future = client.infer_batch(input1_data)
    ...
    # Raises `PyTritonClientTimeoutError` if the server does not respond within 240 seconds
    # from the time `infer_batch` was called by a thread from `ThreadPoolExecutor`
    result_dict = future.result()
```

<!--pytest-codeblocks:cont-->
<!--
```python

# Stop the Triton server to free up resources
triton.stop()
# End of the timeout test

```
-->

!!! warning "gRPC client timeout not fully supported"

    There are some missing features in the gRPC client that prevent it from working correctly with timeouts
    used during the wait for the server and model to be ready. This may cause the client to hang if the server
    doesn't respond with the current server or model state.

!!! info "Server side timeout not implemented"

    Currently, there is no support for server-side timeout. The server will continue to process the request even if the client timeout is reached.