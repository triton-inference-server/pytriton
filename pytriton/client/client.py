# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Clients for easy interaction with models deployed on the Triton Inference Server.

Typical usage example:

    with ModelClient("localhost", "MyModel") as client:
        result_dict = client.infer_sample(input_a=a, input_b=b)

Inference inputs can be provided either as positional or keyword arguments:

    result_dict = client.infer_sample(input1, input2)
    result_dict = client.infer_sample(a=input1, b=input2)

Mixing of argument passing conventions is not supported and will raise PyTritonClientValueError.
"""

import itertools
import logging
import time
import urllib.parse
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tritonclient.grpc
import tritonclient.http
import tritonclient.utils

from pytriton.client.exceptions import (
    PyTritonClientInferenceServerError,
    PyTritonClientModelDoesntSupportBatchingError,
    PyTritonClientUrlParseError,
    PyTritonClientValueError,
)
from pytriton.client.utils import _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S, get_model_config, wait_for_model_ready
from pytriton.constants import DEFAULT_GRPC_PORT, DEFAULT_HTTP_PORT

_LOGGER = logging.getLogger(__name__)

_DEFAULT_INIT_TIMEOUT_S = _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S

_IOType = Union[Tuple[np.ndarray, ...], Dict[str, np.ndarray]]


def _verify_inputs_args(inputs, named_inputs):
    if not inputs and not named_inputs:
        raise PyTritonClientValueError("Provide input data")
    if not bool(inputs) ^ bool(named_inputs):
        raise PyTritonClientValueError("Use either positional either keyword method arguments convention")


class ModelClient:
    """Synchronous client for model deployed on the Triton Inference Server."""

    def __init__(
        self,
        url: str,
        model_name: str,
        model_version: Optional[str] = None,
        *,
        lazy_init: bool = True,
        init_timeout_s: float = _DEFAULT_INIT_TIMEOUT_S,
    ):
        """Inits ModelClient for given model deployed on the Triton Inference Server.

        If `lazy_init` argument is False, model configuration will be read
        from inference server during initialization.

        Common usage:

          with ModelClient("localhost", "BERT") as client
              result_dict = client.infer_sample(input1_sample, input2_sample)

        Args:
            url: The Triton Inference Server url, e.g. 'grpc://localhost:8001'.
                In case no scheme is provided http scheme will be used as default.
                In case no port is provided default port for given scheme will be used -
                8001 for grpc scheme, 8000 for http scheme.
            model_name: name of the model to interact with.
            model_version: version of the model to interact with.
                If model_version is None inference on latest model will be performed.
                The latest versions of the model are numerically the greatest version numbers.
            lazy_init: if initialization should be performed just before sending first request to inference server.
            init_timeout_s: timeout for server and model being ready.

        Raises:
            PyTritonClientModelUnavailableError: If model with given name (and version) is unavailable.
            PyTritonClientTimeoutError:
                if `lazy_init` argument is False and wait time for server and model being ready exceeds `init_timeout_s`.
            PyTritonClientUrlParseError: In case of problems with parsing url.
        """
        if not isinstance(url, str):
            raise PyTritonClientUrlParseError(f"Could not parse url {url}")

        parsed_url = urllib.parse.urlparse(url)
        if not parsed_url.scheme or parsed_url.scheme.lower() not in ["grpc", "http"]:
            _LOGGER.debug(f"Adding http scheme to {url}")
            parsed_url = urllib.parse.urlparse(f"http://{url}")

        port = parsed_url.port or {"grpc": DEFAULT_GRPC_PORT, "http": DEFAULT_HTTP_PORT}[parsed_url.scheme.lower()]
        self._url = f"{parsed_url.hostname}:{port}"
        self._model_name = model_name
        self._model_version = model_version

        self._triton_client_lib = {"grpc": tritonclient.grpc, "http": tritonclient.http}[parsed_url.scheme.lower()]
        _LOGGER.debug(f"Creating InferenceServerClient for {parsed_url.scheme}://{self._url}")
        self._client = self._triton_client_lib.InferenceServerClient(self._url)

        self._request_id_generator = itertools.count(0)
        self._init_timeout_s = init_timeout_s
        self._model_config = None
        self._model_ready = None
        self._lazy_init = lazy_init

        if not self._lazy_init:
            self._wait_and_init_model_config(self._init_timeout_s)

    def __enter__(self):
        """Create context for use _ModelClientBase as a context manager."""
        return self

    def __exit__(self, *_):
        """Close resources used by _ModelClientBase when exiting from context."""
        self.close()

    def close(self):
        """Close resources used by _ModelClientBase."""
        _LOGGER.debug("Closing InferenceServerClient")
        self._client.close()

    def wait_for_model(self, timeout_s: float):
        """Wait for Triton Inference Server and deployed on it model readiness.

        Args:
            timeout_s: timeout to server and model get into readiness state.

        Raises:
            PyTritonClientTimeoutError: If server and model are not in readiness state before given timeout.
            PyTritonClientModelUnavailableError: If model with given name (and version) is unavailable.
            KeyboardInterrupt: If hosting process receives SIGINT
        """
        wait_for_model_ready(self._client, self._model_name, self._model_version, timeout_s=timeout_s)

    @property
    def model_config(self):
        """Obtain configuration of model deployed on the Triton Inference Server.

        Also waits for server to get into readiness state.
        """
        if not self._model_config:
            self._model_config = get_model_config(
                self._client, self._model_name, self._model_version, timeout_s=self._init_timeout_s
            )
        return self._model_config

    def infer_sample(self, *inputs, **named_inputs) -> Dict[str, np.ndarray]:
        """Run synchronous inference on single data sample.

        Typical usage:

            with ModelClient("localhost", "MyModel") as client:
                result_dict = client.infer_sample(input1, input2)

        Inference inputs can be provided either as positional or keyword arguments:

            result_dict = client.infer_sample(input1, input2)
            result_dict = client.infer_sample(a=input1, b=input2)

        Mixing of argument passing conventions is not supported and will raise PyTritonClientRuntimeError.

        Args:
            *inputs: inference inputs provided as positional arguments.
            **named_inputs: inference inputs provided as named arguments.

        Returns:
            dictionary with inference results, where dictionary keys are output names.

        Raises:
            PyTritonClientValueError: if mixing of positional and named arguments passing detected.
            PyTritonClientTimeoutError:
                in case of first method call, `lazy_init` argument is False
                and wait time for server and model being ready exceeds `init_timeout_s`
                or inference time exceeds `timeout_s`.
            PyTritonClientModelUnavailableError: If model with given name (and version) is unavailable.
            PyTritonClientInferenceServerError: If error occurred on inference function or Triton Inference Server side.
        """
        _verify_inputs_args(inputs, named_inputs)

        model_supports_batching = self.model_config.max_batch_size > 0
        if model_supports_batching:
            if inputs:
                inputs = tuple(data[np.newaxis, ...] for data in inputs)
            elif named_inputs:
                named_inputs = {name: data[np.newaxis, ...] for name, data in named_inputs.items()}

        result = self._infer(inputs or named_inputs)
        if model_supports_batching:
            result = {name: data[0] for name, data in result.items()}

        return result

    def infer_batch(self, *inputs, **named_inputs) -> Dict[str, np.ndarray]:
        """Run synchronous inference on batched data.

        Typical usage:

            with ModelClient("localhost", "MyModel") as client:
                result_dict = client.infer_sample(input1, input2)

        Inference inputs can be provided either as positional or keyword arguments:

            result_dict = client.infer_batch(input1, input2)
            result_dict = client.infer_batch(a=input1, b=input2)

        Mixing of argument passing conventions is not supported and will raise PyTritonClientValueError.

        Args:
            *inputs: inference inputs provided as positional arguments.
            **named_inputs: inference inputs provided as named arguments.

        Returns:
            dictionary with inference results, where dictionary keys are output names.

        Raises:
            PyTritonClientValueError: if mixing of positional and named arguments passing detected.
            PyTritonClientTimeoutError:
                in case of first method call, `lazy_init` argument is False
                and wait time for server and model being ready exceeds `init_timeout_s`
                or inference time exceeds `timeout_s`.
            PyTritonClientModelDoesntSupportBatchingError: if model doesn't support batching.
            PyTritonClientModelUnavailableError: If model with given name (and version) is unavailable.
            PyTritonClientInferenceServerError: If error occurred on inference function or Triton Inference Server side.
        """
        _verify_inputs_args(inputs, named_inputs)

        model_supports_batching = self.model_config.max_batch_size > 0
        if not model_supports_batching:
            raise PyTritonClientModelDoesntSupportBatchingError(
                f"Model {self.model_config.model_name} doesn't support batching - use infer_sample method instead"
            )

        return self._infer(inputs or named_inputs)

    def _wait_and_init_model_config(self, init_timeout_s: float):
        """Waits for model and obtain model configuration.

        Args:
            init_timeout_s: timeout for server and model being ready.

        Raises:
            PyTritonClientTimeoutError: if wait time for server and model being ready exceeds `init_timeout_s`
            PyTritonClientModelUnavailableError: If model with given name (and version) is unavailable.
        """
        should_finish_before_s = time.time() + init_timeout_s
        self.wait_for_model(init_timeout_s)
        self._model_ready = True
        timeout_s = max(0.0, should_finish_before_s - time.time())
        self._model_config = get_model_config(self._client, self._model_name, self._model_version, timeout_s=timeout_s)

    def _infer(self, inputs: _IOType) -> Dict[str, np.ndarray]:
        if self._model_ready:
            self._wait_and_init_model_config(self._init_timeout_s)

        if isinstance(inputs, Tuple):
            inputs = {input_spec.name: input_data for input_spec, input_data in zip(self.model_config.inputs, inputs)}

        inputs_wrapped = []

        for input_name, input_data in inputs.items():
            if input_data.dtype == object and not isinstance(input_data.reshape(-1)[0], bytes):
                raise RuntimeError(
                    f"Numpy array for '{input_name}' input with dtype=object should contain encoded strings \
                    \\(e.g. into utf-8\\). Element type: {type(input_data.reshape(-1)[0])}"
                )
            if input_data.dtype.type == np.str_:
                raise RuntimeError(
                    "Unicode inputs are not supported. "
                    f"Encode numpy array for '{input_name}' input (ex. with np.char.encode(array, 'utf-8'))."
                )
            triton_dtype = tritonclient.utils.np_to_triton_dtype(input_data.dtype)
            infer_input = self._triton_client_lib.InferInput(input_name, input_data.shape, triton_dtype)
            infer_input.set_data_from_numpy(input_data)
            inputs_wrapped.append(infer_input)

        outputs_wrapped = [
            self._triton_client_lib.InferRequestedOutput(output_spec.name) for output_spec in self.model_config.outputs
        ]

        try:
            response = self._client.infer(
                model_name=self._model_name,
                model_version=self._model_version or "",
                inputs=inputs_wrapped,
                outputs=outputs_wrapped,
                request_id=str(next(self._request_id_generator)),
            )
        except tritonclient.utils.InferenceServerException as e:
            raise PyTritonClientInferenceServerError(
                f"Error occurred on Triton Inference Server side:\n {e.message()}"
            ) from e

        if isinstance(response, tritonclient.http.InferResult):
            outputs = {
                output["name"]: response.as_numpy(output["name"]) for output in response.get_response()["outputs"]
            }
        else:
            outputs = {output.name: response.as_numpy(output.name) for output in response.get_response().outputs}

        return outputs
