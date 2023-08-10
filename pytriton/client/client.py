# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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

import asyncio
import itertools
import logging
import socket
import sys
import time
import urllib
from concurrent.futures import Future
from queue import Queue
from threading import Thread
from typing import Dict, Optional, Tuple, Union

import async_timeout
import gevent
import numpy as np
import tritonclient.grpc
import tritonclient.grpc.aio
import tritonclient.http
import tritonclient.http.aio
import tritonclient.utils

from pytriton.client.asyncio_utils import asyncio_get_model_config, asyncio_wait_for_model_ready
from pytriton.client.exceptions import (
    PyTritonClientClosedError,
    PyTritonClientInferenceServerError,
    PyTritonClientInvalidUrlError,
    PyTritonClientModelDoesntSupportBatchingError,
    PyTritonClientTimeoutError,
    PyTritonClientValueError,
)
from pytriton.client.utils import (
    _DEFAULT_NETWORK_TIMEOUT_S,
    _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S,
    get_model_config,
    wait_for_model_ready,
)
from pytriton.constants import DEFAULT_GRPC_PORT, DEFAULT_HTTP_PORT
from pytriton.model_config.triton_model_config import TritonModelConfig

_LOGGER = logging.getLogger(__name__)

_DEFAULT_SYNC_INIT_TIMEOUT_S = _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S
_DEFAULT_FUTURES_INIT_TIMEOUT_S = _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S
DEFAULT_INFERENCE_TIMEOUT_S = 60.0


_IOType = Union[Tuple[np.ndarray, ...], Dict[str, np.ndarray]]


def _verify_inputs_args(inputs, named_inputs):
    if not inputs and not named_inputs:
        raise PyTritonClientValueError("Provide input data")
    if not bool(inputs) ^ bool(named_inputs):
        raise PyTritonClientValueError("Use either positional either keyword method arguments convention")


def _verify_parameters(parameters_or_headers: Optional[Dict[str, Union[str, int, bool]]] = None):
    if parameters_or_headers is None:
        return
    if not isinstance(parameters_or_headers, dict):
        raise PyTritonClientValueError("Parameters and headers must be a dictionary")
    for key, value in parameters_or_headers.items():
        if not isinstance(key, str):
            raise PyTritonClientValueError("Parameter/header key must be a string")
        if not isinstance(value, (str, int, bool)):
            raise PyTritonClientValueError("Parameter/header value must be a string, integer or boolean")


class BaseModelClient:
    """Base client for model deployed on the Triton Inference Server."""

    def __init__(
        self,
        url: str,
        model_name: str,
        model_version: Optional[str] = None,
        *,
        lazy_init: bool = True,
        init_timeout_s: Optional[float] = None,
        inference_timeout_s: Optional[float] = None,
    ):
        """Inits BaseModelClient for given model deployed on the Triton Inference Server.

        Common usage:

            ```
            with ModelClient("localhost", "BERT") as client
                result_dict = client.infer_sample(input1_sample, input2_sample)
            ```

        Args:
            url: The Triton Inference Server url, e.g. `grpc://localhost:8001`.
                In case no scheme is provided http scheme will be used as default.
                In case no port is provided default port for given scheme will be used -
                8001 for grpc scheme, 8000 for http scheme.
            model_name: name of the model to interact with.
            model_version: version of the model to interact with.
                If model_version is None inference on latest model will be performed.
                The latest versions of the model are numerically the greatest version numbers.
            lazy_init: if initialization should be performed just before sending first request to inference server.
            init_timeout_s: timeout in seconds for the server and model to be ready. If not passed, the default timeout of 300 seconds will be used.
            inference_timeout_s: timeout in seconds for a single model inference request. If not passed, the default timeout of 60 seconds will be used.

        Raises:
            PyTritonClientModelUnavailableError: If model with given name (and version) is unavailable.
            PyTritonClientTimeoutError:
                if `lazy_init` argument is False and wait time for server and model being ready exceeds `init_timeout_s`.
            PyTritonClientInvalidUrlError: If provided Triton Inference Server url is invalid.
        """
        self._init_timeout_s = _DEFAULT_SYNC_INIT_TIMEOUT_S if init_timeout_s is None else init_timeout_s
        self._inference_timeout_s = DEFAULT_INFERENCE_TIMEOUT_S if inference_timeout_s is None else inference_timeout_s
        self._network_timeout_s = min(_DEFAULT_NETWORK_TIMEOUT_S, self._init_timeout_s)

        self._general_client = self.create_client_from_url(url, network_timeout_s=self._network_timeout_s)
        self._infer_client = self.create_client_from_url(url, network_timeout_s=self._inference_timeout_s)

        self._model_name = model_name
        self._model_version = model_version

        self._request_id_generator = itertools.count(0)

        # Monkey patch __del__ method from client to catch error in client when instance is garbage collected.
        # This is needed because we are closing client in __exit__ method or in close method.
        # (InferenceClient uses gevent library which does not support closing twice from different threads)
        self._monkey_patch_client()

        self._model_config = None
        self._model_ready = None
        self._lazy_init: bool = lazy_init

        self._handle_lazy_init()

    def create_client_from_url(self, url: str, network_timeout_s: Optional[float] = None):
        """Create Triton Inference Server client.

        Args:
            url: url of the server to connect to.
                If url doesn't contain scheme (e.g. "localhost:8001") http scheme is added.
                If url doesn't contain port (e.g. "localhost") default port for given scheme is added.
            network_timeout_s: timeout for client commands. Default value is 60.0 s.

        Returns:
            Triton Inference Server client.

        Raises:
            PyTritonClientInvalidUrlError: If provided Triton Inference Server url is invalid.
        """
        if not isinstance(url, str):
            raise PyTritonClientInvalidUrlError(f"Invalid url {url}. Url must be a string.")

        try:
            parsed_url = urllib.parse.urlparse(url)
            # change in py3.9+
            # https://github.com/python/cpython/commit/5a88d50ff013a64fbdb25b877c87644a9034c969
            if sys.version_info < (3, 9) and not parsed_url.scheme and "://" in parsed_url.path:
                raise ValueError(f"Invalid url {url}. Only grpc and http are supported.")
            if (sys.version_info < (3, 9) and not parsed_url.scheme and "://" not in parsed_url.path) or (
                sys.version_info >= (3, 9) and parsed_url.scheme and not parsed_url.netloc
            ):
                _LOGGER.debug(f"Adding http scheme to {url}")
                parsed_url = urllib.parse.urlparse(f"http://{url}")

            self._scheme = parsed_url.scheme.lower()
            if self._scheme not in ["grpc", "http"]:
                raise ValueError(f"Invalid scheme {self._scheme}. Only grpc and http are supported.")
            port = parsed_url.port or {"grpc": DEFAULT_GRPC_PORT, "http": DEFAULT_HTTP_PORT}[self._scheme]
        except ValueError as e:
            raise PyTritonClientInvalidUrlError(f"Invalid url {url}") from e

        self._triton_client_lib = self.get_lib()

        self._url = f"{parsed_url.hostname}:{port}"
        self._monkey_patch_client()

        if self._scheme == "grpc":
            # by default grpc client has very large number of timeout, thus we want to make it equal to http client timeout
            network_timeout_s = _DEFAULT_NETWORK_TIMEOUT_S if network_timeout_s is None else network_timeout_s
            _LOGGER.warning(
                f"tritonclient.grpc doesn't support timeout for other commands than infer. Ignoring network_timeout: {network_timeout_s}."
            )

        triton_client_init_kwargs = self._get_init_extra_args()

        _LOGGER.debug(
            f"Creating InferenceServerClient for {parsed_url.scheme}://{url} with {triton_client_init_kwargs}"
        )
        return self._triton_client_lib.InferenceServerClient(self._url, **triton_client_init_kwargs)

    def get_lib(self):
        """Returns tritonclient library for given scheme."""
        raise NotImplementedError

    @property
    def _next_request_id(self) -> str:
        # pytype complained about creating generator in __init__ method
        # so we create it lazily
        if getattr(self, "_request_id_generator", None) is None:
            self._request_id_generator = itertools.count(0)
        return str(next(self._request_id_generator))

    def _get_init_extra_args(self):
        timeout = self._inference_timeout_s  # pytype: disable=attribute-error
        #  The inference timeout is used for both the HTTP and the GRPC protocols. However,
        #  the way the timeout is passed to the client differs depending on the protocol.
        #  For the HTTP protocol, the timeout is set in the ``__init__`` method as ``network_timeout``
        #  and ``connection_timeout``. For the GRPC protocol, the timeout
        #  is passed to the infer method as ``client_timeout``.
        #  Both protocols support timeouts correctly and will raise an exception
        #  if the network request or the inference process takes longer than the timeout.
        #  This is a design choice of the underlying tritonclient library.

        if self._scheme != "http":
            return {}

        kwargs = {
            # This value sets the maximum time allowed for each network request in both model loading and inference process
            "network_timeout": timeout,
            # This value sets the maximum time allowed for establishing a connection to the server.
            # We use the inference timeout here instead of the init timeout because the init timeout
            # is meant for waiting for the model to be ready. The connection timeout should be shorter
            # than the init timeout because it only checks if connection is established (e.g. correct port)
            "connection_timeout": timeout,
        }
        return kwargs

    def _monkey_patch_client(self):
        pass

    def _get_model_config_extra_args(self):
        # For the GRPC protocol, the timeout must be passed to the each request as client_timeout
        # model_config doesn't yet support timeout but it is planned for the future
        # grpc_network_timeout_s will be used for model_config
        return {}

    def _handle_lazy_init(self):
        raise NotImplementedError


class ModelClient(BaseModelClient):
    """Synchronous client for model deployed on the Triton Inference Server."""

    def __init__(
        self,
        url: str,
        model_name: str,
        model_version: Optional[str] = None,
        *,
        lazy_init: bool = True,
        init_timeout_s: Optional[float] = None,
        inference_timeout_s: Optional[float] = None,
    ):
        """Inits ModelClient for given model deployed on the Triton Inference Server.

        If `lazy_init` argument is False, model configuration will be read
        from inference server during initialization.

        Common usage:
        ```
        with ModelClient("localhost", "BERT") as client
            result_dict = client.infer_sample(input1_sample, input2_sample)
        ```

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
            init_timeout_s: timeout for maximum waiting time in loop, which sends retry requests ask if model is ready. It is applied at initialization time only when `lazy_init` argument is False. Default is to do retry loop at first inference.
            inference_timeout_s: timeout in seconds for the model inference process.
                If non passed default 60 seconds timeout will be used.
                For HTTP client it is not only inference timeout but any client request timeout
                - get model config, is model loaded. For GRPC client it is only inference timeout.

        Raises:
            PyTritonClientModelUnavailableError: If model with given name (and version) is unavailable.
            PyTritonClientTimeoutError:
                if `lazy_init` argument is False and wait time for server and model being ready exceeds `init_timeout_s`.
            PyTritonClientUrlParseError: In case of problems with parsing url.
        """
        super().__init__(
            url=url,
            model_name=model_name,
            model_version=model_version,
            lazy_init=lazy_init,
            init_timeout_s=init_timeout_s,
            inference_timeout_s=inference_timeout_s,
        )

    def get_lib(self):
        """Returns tritonclient library for given scheme."""
        return {"grpc": tritonclient.grpc, "http": tritonclient.http}[self._scheme.lower()]

    def __enter__(self):
        """Create context for using ModelClient as a context manager."""
        return self

    def __exit__(self, *_):
        """Close resources used by ModelClient instance when exiting from the context."""
        self.close()

    def close(self):
        """Close resources used by ModelClient.

        This method closes the resources used by the ModelClient instance,
        including the Triton Inference Server connections.
        Once this method is called, the ModelClient instance should not be used again.
        """
        _LOGGER.debug("Closing ModelClient")
        try:
            if self._general_client is not None:
                self._general_client.close()
            if self._infer_client is not None:
                self._infer_client.close()
            self._general_client = None
            self._infer_client = None
        except Exception as e:
            _LOGGER.warning("Error while closing ModelClient resources: %s", e)
            raise e

    def wait_for_model(self, timeout_s: float):
        """Wait for the Triton Inference Server and the deployed model to be ready.

        Args:
            timeout_s: timeout in seconds to wait for the server and model to be ready.

        Raises:
            PyTritonClientTimeoutError: If the server and model are not ready before the given timeout.
            PyTritonClientModelUnavailableError: If the model with the given name (and version) is unavailable.
            KeyboardInterrupt: If the hosting process receives SIGINT.
            PyTritonClientClosedError: If the ModelClient is closed.
        """
        if self._general_client is None:
            raise PyTritonClientClosedError("ModelClient is closed")
        wait_for_model_ready(self._general_client, self._model_name, self._model_version, timeout_s=timeout_s)

    @property
    def is_batching_supported(self):
        """Checks if model supports batching.

        Also waits for server to get into readiness state.
        """
        return self.model_config.max_batch_size > 0

    @property
    def model_config(self) -> TritonModelConfig:
        """Obtain the configuration of the model deployed on the Triton Inference Server.

        This method waits for the server to get into readiness state before obtaining the model configuration.

        Returns:
            TritonModelConfig: configuration of the model deployed on the Triton Inference Server.

        Raises:
            PyTritonClientTimeoutError: If the server and model are not in readiness state before the given timeout.
            PyTritonClientModelUnavailableError: If the model with the given name (and version) is unavailable.
            KeyboardInterrupt: If the hosting process receives SIGINT.
            PyTritonClientClosedError: If the ModelClient is closed.
        """
        if not self._model_config:
            if self._general_client is None:
                raise PyTritonClientClosedError("ModelClient is closed")

            self._model_config = get_model_config(
                self._general_client, self._model_name, self._model_version, timeout_s=self._init_timeout_s
            )
        return self._model_config

    def infer_sample(
        self,
        *inputs,
        parameters: Optional[Dict[str, Union[str, int, bool]]] = None,
        headers: Optional[Dict[str, Union[str, int, bool]]] = None,
        **named_inputs,
    ) -> Dict[str, np.ndarray]:
        """Run synchronous inference on a single data sample.

        Typical usage:

            ```python
            with ModelClient("localhost", "MyModel") as client:
                result_dict = client.infer_sample(input1, input2)
            ```
        Inference inputs can be provided either as positional or keyword arguments:

            ```python
            result_dict = client.infer_sample(input1, input2)
            result_dict = client.infer_sample(a=input1, b=input2)
            ```

        Args:
            *inputs: Inference inputs provided as positional arguments.
            parameters: Custom inference parameters.
            headers: Custom inference headers.
            **named_inputs: Inference inputs provided as named arguments.

        Returns:
            Dictionary with inference results, where dictionary keys are output names.

        Raises:
            PyTritonClientValueError: If mixing of positional and named arguments passing detected.
            PyTritonClientTimeoutError: If the wait time for the server and model being ready exceeds `init_timeout_s` or
                inference request time exceeds `inference_timeout_s`.
            PyTritonClientModelUnavailableError: If the model with the given name (and version) is unavailable.
            PyTritonClientInferenceServerError: If an error occurred on the inference callable or Triton Inference Server side.
        """
        _verify_inputs_args(inputs, named_inputs)
        _verify_parameters(parameters)
        _verify_parameters(headers)

        if self.is_batching_supported:
            if inputs:
                inputs = tuple(data[np.newaxis, ...] for data in inputs)
            elif named_inputs:
                named_inputs = {name: data[np.newaxis, ...] for name, data in named_inputs.items()}

        result = self._infer(inputs or named_inputs, parameters, headers)
        if self.is_batching_supported:
            result = {name: data[0] for name, data in result.items()}

        return result

    def infer_batch(
        self,
        *inputs,
        parameters: Optional[Dict[str, Union[str, int, bool]]] = None,
        headers: Optional[Dict[str, Union[str, int, bool]]] = None,
        **named_inputs,
    ) -> Dict[str, np.ndarray]:
        """Run synchronous inference on batched data.

        Typical usage:

            ```python
            with ModelClient("localhost", "MyModel") as client:
                result_dict = client.infer_batch(input1, input2)
            ```

        Inference inputs can be provided either as positional or keyword arguments:

            ```python
            result_dict = client.infer_batch(input1, input2)
            result_dict = client.infer_batch(a=input1, b=input2)
            ```

        Args:
            *inputs: Inference inputs provided as positional arguments.
            parameters: Custom inference parameters.
            headers: Custom inference headers.
            **named_inputs: Inference inputs provided as named arguments.

        Returns:
            Dictionary with inference results, where dictionary keys are output names.

        Raises:
            PyTritonClientValueError: If mixing of positional and named arguments passing detected.
            PyTritonClientTimeoutError: If the wait time for the server and model being ready exceeds `init_timeout_s` or
                inference request time exceeds `inference_timeout_s`.
            PyTritonClientModelUnavailableError: If the model with the given name (and version) is unavailable.
            PyTritonClientInferenceServerError: If an error occurred on the inference callable or Triton Inference Server side.
            PyTritonClientModelDoesntSupportBatchingError: If the model doesn't support batching.
            PyTritonClientValueError: if mixing of positional and named arguments passing detected.
            PyTritonClientTimeoutError:
                in case of first method call, `lazy_init` argument is False
                and wait time for server and model being ready exceeds `init_timeout_s` or
                inference time exceeds `inference_timeout_s` passed to `__init__`.
            PyTritonClientModelUnavailableError: If model with given name (and version) is unavailable.
            PyTritonClientInferenceServerError: If error occurred on inference callable or Triton Inference Server side,
        """
        _verify_inputs_args(inputs, named_inputs)
        _verify_parameters(parameters)
        _verify_parameters(headers)

        if not self.is_batching_supported:
            raise PyTritonClientModelDoesntSupportBatchingError(
                f"Model {self.model_config.model_name} doesn't support batching - use infer_sample method instead"
            )

        return self._infer(inputs or named_inputs, parameters, headers)

    def _wait_and_init_model_config(self, init_timeout_s: float):
        if self._general_client is None:
            raise PyTritonClientClosedError("ModelClient is closed")

        should_finish_before_s = time.time() + init_timeout_s
        self.wait_for_model(init_timeout_s)
        self._model_ready = True
        timeout_s = max(0.0, should_finish_before_s - time.time())
        self._model_config = get_model_config(
            self._general_client, self._model_name, self._model_version, timeout_s=timeout_s
        )

    def _infer(self, inputs: _IOType, parameters, headers) -> Dict[str, np.ndarray]:
        if self._infer_client is None:
            raise PyTritonClientClosedError("ModelClient is closed")

        if not self._model_ready:
            self._wait_and_init_model_config(self._init_timeout_s)

        if isinstance(inputs, Tuple):
            inputs = {input_spec.name: input_data for input_spec, input_data in zip(self.model_config.inputs, inputs)}

        inputs_wrapped = []

        for input_name, input_data in inputs.items():
            if input_data.dtype == object and not isinstance(input_data.reshape(-1)[0], bytes):
                raise RuntimeError(
                    f"Numpy array for {input_name!r} input with dtype=object should contain encoded strings \
                    \\(e.g. into utf-8\\). Element type: {type(input_data.reshape(-1)[0])}"
                )
            if input_data.dtype.type == np.str_:
                raise RuntimeError(
                    "Unicode inputs are not supported. "
                    f"Encode numpy array for {input_name!r} input (ex. with np.char.encode(array, 'utf-8'))."
                )
            triton_dtype = tritonclient.utils.np_to_triton_dtype(input_data.dtype)
            infer_input = self._triton_client_lib.InferInput(input_name, input_data.shape, triton_dtype)
            infer_input.set_data_from_numpy(input_data)
            inputs_wrapped.append(infer_input)

        outputs_wrapped = [
            self._triton_client_lib.InferRequestedOutput(output_spec.name) for output_spec in self.model_config.outputs
        ]

        try:
            _LOGGER.debug("Sending inference request to Triton Inference Server")
            response = self._infer_client.infer(
                model_name=self._model_name,
                model_version=self._model_version or "",
                inputs=inputs_wrapped,
                headers=headers,
                outputs=outputs_wrapped,
                request_id=self._next_request_id,
                parameters=parameters,
                **self._get_infer_extra_args(),
            )
        except tritonclient.utils.InferenceServerException as e:
            # tritonclient.grpc raises execption with message containing "Deadline Exceeded" for timeout
            if "Deadline Exceeded" in e.message():
                raise PyTritonClientTimeoutError(
                    f"Timeout occurred during inference request. Timeout: {self._inference_timeout_s} s. Message: {e.message()}"
                ) from e

            raise PyTritonClientInferenceServerError(
                f"Error occurred during inference request. Message: {e.message()}"
            ) from e
        except socket.timeout as e:  # tritonclient.http raises socket.timeout for timeout
            message = f"Timeout occurred during inference request. Timeout: {self._inference_timeout_s} s Message: {e}"
            _LOGGER.error(message)
            raise PyTritonClientTimeoutError(message) from e
        except OSError as e:  # tritonclient.http raises socket.error for connection error
            message = f"Timeout occurred during inference request. Timeout: {self._inference_timeout_s} s Message: {e}"
            _LOGGER.error(message)
            raise PyTritonClientTimeoutError(message) from e

        if isinstance(response, tritonclient.http.InferResult):
            outputs = {
                output["name"]: response.as_numpy(output["name"]) for output in response.get_response()["outputs"]
            }
        else:
            outputs = {output.name: response.as_numpy(output.name) for output in response.get_response().outputs}

        return outputs

    def _handle_lazy_init(self):
        if not self._lazy_init:
            self._wait_and_init_model_config(self._init_timeout_s)

    def _get_infer_extra_args(self):
        if self._scheme == "http":
            return {}
        # For the GRPC protocol, the timeout is passed to the infer method as client_timeout
        # This timeout applies to the whole inference process and each network request

        # The ``infer`` supports also timeout argument for both GRPC and HTTP.
        # It is applied at server side and supported only for dynamic batching.
        # However, it is not used here yet and planned for future release
        kwargs = {"client_timeout": self._inference_timeout_s}
        return kwargs


class AsyncioModelClient(BaseModelClient):
    """Asyncio client for model deployed on the Triton Inference Server.

    This client is based on Triton Inference Server Python clients and GRPC library:
    * ``tritonclient.http.aio.InferenceServerClient``
    * ``tritonclient.grpc.aio.InferenceServerClient``

    It can wait for server to be ready with model loaded and then perform inference on it.
    ``AsyncioModelClient`` supports asyncio context manager protocol.

    Typical usage:
    ```python
    from pytriton.client import AsyncioModelClient
    import numpy as np

    input1_sample = np.random.rand(1, 3, 224, 224).astype(np.float32)
    input2_sample = np.random.rand(1, 3, 224, 224).astype(np.float32)

    async with AsyncioModelClient("localhost", "MyModel") as client:
        result_dict = await client.infer_sample(input1_sample, input2_sample)
        print(result_dict["output_name"])
    ```
    """

    def __init__(
        self,
        url: str,
        model_name: str,
        model_version: Optional[str] = None,
        *,
        lazy_init: bool = True,
        init_timeout_s: Optional[float] = None,
        inference_timeout_s: Optional[float] = None,
    ):
        """Inits ModelClient for given model deployed on the Triton Inference Server.

        If `lazy_init` argument is False, model configuration will be read
        from inference server during initialization.

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
            PyTritonClientTimeoutError: if `lazy_init` argument is False and wait time for server and model being ready exceeds `init_timeout_s`.
            PyTritonClientUrlParseError: In case of problems with parsing url.
        """
        super().__init__(
            url=url,
            model_name=model_name,
            model_version=model_version,
            lazy_init=lazy_init,
            init_timeout_s=init_timeout_s,
            inference_timeout_s=inference_timeout_s,
        )

    def get_lib(self):
        """Get Triton Inference Server Python client library."""
        return {"grpc": tritonclient.grpc.aio, "http": tritonclient.http.aio}[self._scheme.lower()]

    async def __aenter__(self):
        """Create context for use AsyncioModelClient as a context manager."""
        _LOGGER.debug("Entering AsyncioModelClient context")
        try:
            if not self._lazy_init:
                _LOGGER.debug("Waiting in AsyncioModelClient context for model to be ready")
                await self._wait_and_init_model_config(self._init_timeout_s)
                _LOGGER.debug("Model is ready in AsyncioModelClient context")
            return self
        except Exception as e:
            _LOGGER.error("Error occurred during AsyncioModelClient context initialization")
            await self.close()
            raise e

    async def __aexit__(self, *_):
        """Close resources used by AsyncioModelClient when exiting from context."""
        await self.close()
        _LOGGER.debug("Exiting AsyncioModelClient context")

    async def close(self):
        """Close resources used by _ModelClientBase."""
        _LOGGER.debug("Closing InferenceServerClient")
        await self._general_client.close()
        await self._infer_client.close()
        _LOGGER.debug("InferenceServerClient closed")

    async def wait_for_model(self, timeout_s: float):
        """Asynchronous wait for Triton Inference Server and deployed on it model readiness.

        Args:
            timeout_s: timeout to server and model get into readiness state.

        Raises:
            PyTritonClientTimeoutError: If server and model are not in readiness state before given timeout.
            PyTritonClientModelUnavailableError: If model with given name (and version) is unavailable.
            KeyboardInterrupt: If hosting process receives SIGINT
        """
        _LOGGER.debug(f"Waiting for model {self._model_name} to be ready")
        async with async_timeout.timeout(self._init_timeout_s):
            await asyncio_wait_for_model_ready(
                self._general_client, self._model_name, self._model_version, timeout_s=timeout_s
            )
        _LOGGER.debug(f"Model {self._model_name} is ready")

    @property
    async def model_config(self):
        """Obtain configuration of model deployed on the Triton Inference Server.

        Also waits for server to get into readiness state.
        """
        try:
            async with async_timeout.timeout(self._init_timeout_s):
                if not self._model_config:
                    kwargs = self._get_model_config_extra_args()
                    _LOGGER.debug(f"Obtaining model config for {self._model_name}")
                    self._model_config = await asyncio_get_model_config(
                        self._general_client,
                        self._model_name,
                        self._model_version,
                        timeout_s=self._init_timeout_s,
                        **kwargs,
                    )
                    _LOGGER.debug(f"Obtained model config for {self._model_name}")
                return self._model_config
        except asyncio.TimeoutError as e:
            message = f"Timeout while waiting for model {self._model_name} to be ready for {self._init_timeout_s}s"
            _LOGGER.error(message)
            raise PyTritonClientTimeoutError(message) from e

    async def infer_sample(
        self,
        *inputs,
        parameters: Optional[Dict[str, Union[str, int, bool]]] = None,
        headers: Optional[Dict[str, Union[str, int, bool]]] = None,
        **named_inputs,
    ) -> Dict[str, np.ndarray]:
        """Run asynchronous inference on single data sample.

        Typical usage:

        ```python
        async with AsyncioModelClient("localhost", "MyModel") as client:
            result_dict = await client.infer_sample(input1, input2)
        ```

        Inference inputs can be provided either as positional or keyword arguments:

        ```python
        result_dict = await client.infer_sample(input1, input2)
        result_dict = await client.infer_sample(a=input1, b=input2)
        ```

        Mixing of argument passing conventions is not supported and will raise PyTritonClientRuntimeError.

        Args:
            *inputs: inference inputs provided as positional arguments.
            parameters: custom inference parameters.
            headers: custom inference headers.
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
            PyTritonClientInferenceServerError: If error occurred on inference callable or Triton Inference Server side.
        """
        _verify_inputs_args(inputs, named_inputs)
        _verify_parameters(parameters)
        _verify_parameters(headers)

        _LOGGER.debug(f"Running inference for {self._model_name}")
        model_config = await self.model_config
        _LOGGER.debug(f"Model config for {self._model_name} obtained")

        model_supports_batching = model_config.max_batch_size > 0
        if model_supports_batching:
            if inputs:
                inputs = tuple(data[np.newaxis, ...] for data in inputs)
            elif named_inputs:
                named_inputs = {name: data[np.newaxis, ...] for name, data in named_inputs.items()}

        _LOGGER.debug(f"Running _infer for {self._model_name}")
        result = await self._infer(inputs or named_inputs, parameters, headers)
        _LOGGER.debug(f"_infer for {self._model_name} finished")
        if model_supports_batching:
            result = {name: data[0] for name, data in result.items()}

        return result

    async def infer_batch(
        self,
        *inputs,
        parameters: Optional[Dict[str, Union[str, int, bool]]] = None,
        headers: Optional[Dict[str, Union[str, int, bool]]] = None,
        **named_inputs,
    ) -> Dict[str, np.ndarray]:
        """Run asynchronous inference on batched data.

        Typical usage:

        ```python
        async with AsyncioModelClient("localhost", "MyModel") as client:
            result_dict = await client.infer_batch(input1, input2)
        ```

        Inference inputs can be provided either as positional or keyword arguments:

        ```python
        result_dict = await client.infer_batch(input1, input2)
        result_dict = await client.infer_batch(a=input1, b=input2)
        ```

        Mixing of argument passing conventions is not supported and will raise PyTritonClientValueError.

        Args:
            *inputs: inference inputs provided as positional arguments.
            parameters: custom inference parameters.
            headers: custom inference headers.
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
            PyTritonClientInferenceServerError: If error occurred on inference callable or Triton Inference Server side.
        """
        _verify_inputs_args(inputs, named_inputs)
        _verify_parameters(parameters)
        _verify_parameters(headers)

        _LOGGER.debug(f"Running inference for {self._model_name}")
        model_config = await self.model_config
        _LOGGER.debug(f"Model config for {self._model_name} obtained")

        model_supports_batching = model_config.max_batch_size > 0
        if not model_supports_batching:
            _LOGGER.error(f"Model {model_config.model_name} doesn't support batching")
            raise PyTritonClientModelDoesntSupportBatchingError(
                f"Model {model_config.model_name} doesn't support batching - use infer_sample method instead"
            )

        _LOGGER.debug(f"Running _infer for {self._model_name}")
        result = await self._infer(inputs or named_inputs, parameters, headers)
        _LOGGER.debug(f"_infer for {self._model_name} finished")
        return result

    async def _wait_and_init_model_config(self, init_timeout_s: float):
        """Asynchronous wait for model and obtain model configuration.

        Args:
            init_timeout_s: timeout for server and model being ready.

        Raises:
            PyTritonClientTimeoutError: if wait time for server and model being ready exceeds `init_timeout_s`
            PyTritonClientModelUnavailableError: If model with given name (and version) is unavailable.
        """
        try:
            async with async_timeout.timeout(init_timeout_s):
                should_finish_before_s = time.time() + init_timeout_s
                _LOGGER.debug(f"Waiting for model {self._model_name} to be ready")
                await self.wait_for_model(init_timeout_s)
                _LOGGER.debug(f"Model {self._model_name} is ready")
                self._model_ready = True
                timeout_s = max(0.0, should_finish_before_s - time.time())
                _LOGGER.debug(f"Obtaining model config for {self._model_name}")
                self._model_config = await asyncio_get_model_config(
                    self._general_client, self._model_name, self._model_version, timeout_s=timeout_s
                )
                _LOGGER.debug(f"Model config for {self._model_name} obtained")
        except asyncio.TimeoutError as e:
            _LOGGER.error(f"Timeout exceeded while waiting for model {self._model_name} to be ready")
            raise PyTritonClientTimeoutError(
                f"Timeout exceeded while waiting for model {self._model_name} to be ready"
            ) from e

    def _validate_input(self, input_name, input_data):
        if input_data.dtype == object and not isinstance(input_data.reshape(-1)[0], bytes):
            raise RuntimeError(
                f"Numpy array for {input_name!r} input with dtype=object should contain encoded strings \
                \\(e.g. into utf-8\\). Element type: {type(input_data.reshape(-1)[0])}"
            )
        if input_data.dtype.type == np.str_:
            raise RuntimeError(
                "Unicode inputs are not supported. "
                f"Encode numpy array for {input_name!r} input (ex. with np.char.encode(array, 'utf-8'))."
            )

    async def _execute_infer(self, model_config, inputs_wrapped, outputs_wrapped, parameters, headers):
        try:
            _LOGGER.debug(f"Sending InferRequest for {self._model_name}")
            kwargs = self._get_infer_extra_args()
            response = await self._infer_client.infer(
                model_name=self._model_name,
                model_version=self._model_version or "",
                inputs=inputs_wrapped,
                headers=headers,
                outputs=outputs_wrapped,
                request_id=self._next_request_id,
                parameters=parameters,
                **kwargs,
            )
        except asyncio.exceptions.TimeoutError as e:
            # HTTP aio client raises asyncio.exceptions.TimeoutError for timeout errors
            message = f"Timeout exceeded while running inference for {self._model_name}"
            _LOGGER.error(message)
            raise PyTritonClientTimeoutError(message) from e
        except tritonclient.utils.InferenceServerException as e:
            message = f"Error occurred on Triton Inference Server side:\n {e.message()}"
            _LOGGER.error(message)
            if "Deadline Exceeded" in e.message():
                # GRPC aio client raises InferenceServerException with message "Deadline Exceeded"
                # for timeout errors
                raise PyTritonClientTimeoutError(message) from e
            else:
                raise PyTritonClientInferenceServerError(message) from e
        _LOGGER.debug(f"Received InferResponse for {self._model_name}")
        outputs = {output_spec.name: response.as_numpy(output_spec.name) for output_spec in model_config.outputs}
        return outputs

    async def _infer(self, inputs: _IOType, parameters, headers) -> Dict[str, np.ndarray]:
        if self._model_ready:
            _LOGGER.debug(f"Waiting for model {self._model_name} config")
            await self._wait_and_init_model_config(self._init_timeout_s)
            _LOGGER.debug(f"Model wait finished for {self._model_name}")

        _LOGGER.debug(f"Obtaining config for {self._model_name}")
        model_config = await self.model_config
        _LOGGER.debug(f"Model config for {self._model_name} obtained")

        if isinstance(inputs, Tuple):
            inputs = {input_spec.name: input_data for input_spec, input_data in zip(model_config.inputs, inputs)}

        inputs_wrapped = []
        for input_name, input_data in inputs.items():
            if isinstance(input_data, np.ndarray):
                self._validate_input(input_name, input_data)
                triton_dtype = tritonclient.utils.np_to_triton_dtype(input_data.dtype)
                infer_input = self._triton_client_lib.InferInput(input_name, input_data.shape, triton_dtype)
                infer_input.set_data_from_numpy(input_data)
                input_wrapped = infer_input
                inputs_wrapped.append(input_wrapped)
            else:
                raise PyTritonClientValueError(
                    f"Input {input_name} is not a numpy array. Got {type(input_data)} instead."
                )

        outputs_wrapped = [
            self._triton_client_lib.InferRequestedOutput(output_spec.name) for output_spec in model_config.outputs
        ]
        return await self._execute_infer(model_config, inputs_wrapped, outputs_wrapped, parameters, headers)

    def _handle_lazy_init(self):
        # Asynchronous lazy initialization is done in __aenter__ method
        pass

    def _get_init_extra_args(self):
        #  The inference timeout is used for both the HTTP and the GRPC protocols. However,
        #  the way the timeout is passed to the client differs depending on the protocol.
        #  For the HTTP protocol, the timeout is set in the ``__init__`` method as ``conn_timeout`` for both connection and request timeouts.
        #  For the GRPC protocol, the timeout
        #  is passed to the infer method as ``client_timeout``.
        #  Both protocols support timeouts correctly and will raise an exception
        #  if the network request or the inference process takes longer than the timeout.
        #  This is a design choice of the underlying tritonclient library.

        if self._scheme != "http":
            return {}

        kwargs = {
            # This value sets the maximum time allowed for both connection and network requests in both model loading and inference process
            "conn_timeout": self._inference_timeout_s,
        }
        return kwargs

    def _get_infer_extra_args(self):
        if self._scheme == "http":
            return {}
        # For the GRPC protocol, the timeout is passed to the infer method as client_timeout
        # This timeout applies to the whole inference process and each network request

        # The ``infer`` supports also timeout argument for both GRPC and HTTP.
        # It is applied at server side and supported only for dynamic batching.
        # However, it is not used here yet and planned for future release
        kwargs = {"client_timeout": self._inference_timeout_s}
        return kwargs


_INIT = "init"
_WAIT_FOR_MODEL = "wait_for_model"
_MODEL_CONFIG = "model_config"
_INFER_BATCH = "infer_batch"
_INFER_SAMPLE = "infer_sample"
_CLOSE = "close"


class FuturesModelClient:
    """A client for interacting with a model deployed on the Triton Inference Server using concurrent.futures.

    This client allows asynchronous inference requests using a thread pool executor. It can be used to perform inference
    on a model by providing input data and receiving the corresponding output data. The client can be used in a `with`
    statement to ensure proper resource management.

    Example usage:

        ```python
        with FuturesModelClient("localhost", "MyModel") as client:
            result_future = client.infer_sample(input1=input1_data, input2=input2_data)
            # do something else
            print(result_future.result())
        ```
    """

    def __init__(
        self,
        url: str,
        model_name: str,
        model_version: Optional[str] = None,
        *,
        max_workers: Optional[int] = None,
        init_timeout_s: Optional[float] = None,
        inference_timeout_s: Optional[float] = None,
    ):
        """Initializes the FuturesModelClient for a given model.

        Args:
            url: The Triton Inference Server url, e.g. `grpc://localhost:8001`.
            model_name: The name of the model to interact with.
            model_version: The version of the model to interact with. If None, the latest version will be used.
            max_workers: The maximum number of threads that can be used to execute the given calls. If None, the `min(32, os.cpu_count() + 4)` number of threads will be used.
            init_timeout_s: Timeout in seconds for server and model being ready. If non passed default 60 seconds timeout will be used.
            inference_timeout_s: Timeout in seconds for the single model inference request. If non passed default 60 seconds timeout will be used.
        """
        self._url = url
        self._model_name = model_name
        self._model_version = model_version
        self._threads = []
        self._max_workers = max_workers
        if self._max_workers is not None and self._max_workers <= 0:
            raise ValueError("max_workers must be greater than 0")
        kwargs = {}
        if self._max_workers is not None:
            kwargs["maxsize"] = self._max_workers
        self._queue = Queue(**kwargs)
        self._queue.put((_INIT, None, None))
        self._init_timeout_s = _DEFAULT_FUTURES_INIT_TIMEOUT_S if init_timeout_s is None else init_timeout_s
        self._inference_timeout_s = inference_timeout_s
        self._closed = False

    def __enter__(self):
        """Create context for using FuturesModelClient as a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close resources used by FuturesModelClient instance when exiting from the context."""
        self.close()

    def close(self, wait=True):
        """Close resources used by FuturesModelClient.

        This method closes the resources used by the FuturesModelClient instance, including the Triton Inference Server connections.
        Once this method is called, the FuturesModelClient instance should not be used again.

        Args:
            wait: If True, then shutdown will not return until all running futures have finished executing.
        """
        if self._closed:
            _LOGGER.warning("FuturesModelClient is already closed")
            return
        _LOGGER.debug("Closing FuturesModelClient.")

        self._closed = True
        for _ in range(len(self._threads)):
            self._queue.put((_CLOSE, None, None))

        if wait:
            _LOGGER.debug("Waiting for futures to finish.")
            for thread in self._threads:
                thread.join()

    def wait_for_model(self, timeout_s: float) -> Future:
        """Returns a Future object which result will be None when the model is ready.

        Typical usage:

            ```python
            with FuturesModelClient("localhost", "BERT") as client
                future = client.wait_for_model(300.)
                # do something else
                future.result()   # wait rest of timeout_s time
                                  # till return None if model is ready
                                  # or raise PyTritonClientTimeutError
            ```

        Args:
            timeout_s: The maximum amount of time to wait for the model to be ready, in seconds.

        Returns:
            A Future object which result is None when the model is ready.
        """
        return self._execute(
            name=_WAIT_FOR_MODEL,
            request=timeout_s,
        )

    def model_config(self) -> Future:
        """Obtain the configuration of the model deployed on the Triton Inference Server.

        This method returns a Future object that will contain the TritonModelConfig object when it is ready.
        Client will wait init_timeout_s for the server to get into readiness state before obtaining the model configuration.

        Returns:
            A Future object that will contain the TritonModelConfig object when it is ready.

        Raises:
            PyTritonClientClosedError: If the FuturesModelClient is closed.
        """
        return self._execute(name=_MODEL_CONFIG)

    def infer_sample(
        self,
        *inputs,
        parameters: Optional[Dict[str, Union[str, int, bool]]] = None,
        headers: Optional[Dict[str, Union[str, int, bool]]] = None,
        **named_inputs,
    ) -> Future:
        """Run asynchronous inference on a single data sample and return a Future object.

        This method allows the user to perform inference on a single data sample by providing input data and receiving the
        corresponding output data. The method returns a Future object that wraps a dictionary of inference results, where dictionary keys are output names.

        Example usage:

            ```python
            with FuturesModelClient("localhost", "BERT") as client:
                result_future = client.infer_sample(input1=input1_data, input2=input2_data)
                # do something else
                print(result_future.result())
            ```

        Inference inputs can be provided either as positional or keyword arguments:

            ```python
            future = client.infer_sample(input1, input2)
            future = client.infer_sample(a=input1, b=input2)
            ```

        Args:
            *inputs: Inference inputs provided as positional arguments.
            parameters: Optional dictionary of inference parameters.
            headers: Optional dictionary of HTTP headers for the inference request.
            **named_inputs: Inference inputs provided as named arguments.

        Returns:
            A Future object wrapping a dictionary of inference results, where dictionary keys are output names.

        Raises:
            PyTritonClientClosedError: If the FuturesModelClient is closed.
        """
        return self._execute(
            name=_INFER_SAMPLE,
            request=(inputs, parameters, headers, named_inputs),
        )

    def infer_batch(
        self,
        *inputs,
        parameters: Optional[Dict[str, Union[str, int, bool]]] = None,
        headers: Optional[Dict[str, Union[str, int, bool]]] = None,
        **named_inputs,
    ) -> Future:
        """Run asynchronous inference on batched data and return a Future object.

        This method allows the user to perform inference on batched data by providing input data and receiving the corresponding output data.
        The method returns a Future object that wraps a dictionary of inference results, where dictionary keys are output names.

        Example usage:

            ```python
            with FuturesModelClient("localhost", "BERT") as client:
                future = client.infer_batch(input1_sample, input2_sample)
                # do something else
                print(future.result())
            ```

        Inference inputs can be provided either as positional or keyword arguments:

            ```python
            future = client.infer_batch(input1, input2)
            future = client.infer_batch(a=input1, b=input2)
            ```

        Mixing of argument passing conventions is not supported and will raise PyTritonClientValueError.

        Args:
            *inputs: Inference inputs provided as positional arguments.
            parameters: Optional dictionary of inference parameters.
            headers: Optional dictionary of HTTP headers for the inference request.
            **named_inputs: Inference inputs provided as named arguments.

        Returns:
            A Future object wrapping a dictionary of inference results, where dictionary keys are output names.

        Raises:
            PyTritonClientClosedError: If the FuturesModelClient is closed.
        """
        return self._execute(name=_INFER_BATCH, request=(inputs, parameters, headers, named_inputs))

    def _execute(self, name, request=None):
        if self._closed:
            raise PyTritonClientClosedError("FutureModelClient is already closed")
        self._extend_thread_pool()
        future = Future()
        self._queue.put((future, request, name))
        return future

    def _extend_thread_pool(self):
        if self._closed:
            return
        if not self._queue.empty() and (self._max_workers is None or len(self._threads) < self._max_workers):
            _LOGGER.debug("Create new thread")
            thread = Thread(target=self._worker)
            self._threads.append(thread)
            thread.start()
        else:
            _LOGGER.debug("No need to create new thread")

    def _client_request_executor(self, client, request, name):
        _LOGGER.debug(f"Running {name} for {self._model_name}")
        if name == _INFER_SAMPLE:
            inputs, parameters, headers, named_inputs = request
            return client.infer_sample(
                *inputs,
                parameters=parameters,
                headers=headers,
                **named_inputs,
            )
        elif name == _INFER_BATCH:
            inputs, parameters, headers, named_inputs = request
            return client.infer_batch(
                *inputs,
                parameters=parameters,
                headers=headers,
                **named_inputs,
            )
        elif name == _MODEL_CONFIG:
            return client.model_config
        elif name == _WAIT_FOR_MODEL:
            timeout_s = request
            return client.wait_for_model(timeout_s)
        else:
            raise PyTritonClientValueError(f"Unknown request name {name}")

    def _create_client(self, lazy_init):
        _LOGGER.debug(f"Creating ModelClient lazy_init={lazy_init}")
        return ModelClient(
            self._url,
            self._model_name,
            self._model_version,
            lazy_init=lazy_init,
            init_timeout_s=self._init_timeout_s,
            inference_timeout_s=self._inference_timeout_s,
        )

    def _worker(self):
        _LOGGER.debug("Starting worker thread")
        # Work around for AttributeError: '_Threadlocal' object has no attribute 'hub'
        # gevent/_hub_local.py", line 77, in gevent._gevent_c_hub_local.get_hub_noargs
        gevent.get_hub()
        while True:
            future, request, name = self._queue.get()
            if future == _CLOSE:
                _LOGGER.debug("Closing thread")
                self._queue.task_done()
                break
            if future == _INIT:
                continue
            try:
                client = self._create_client(name == _WAIT_FOR_MODEL)
                with client:
                    while True:
                        try:
                            result = self._client_request_executor(client, request, name)
                            _LOGGER.debug(f"Finished {name} for {self._model_name}")
                            future.set_result(result)
                            self._queue.task_done()
                        except Exception as e:
                            _LOGGER.error(f"Error {e} occurred during {name} for {self._model_name}")
                            future.set_exception(e)
                            self._queue.task_done()
                            break
                        future, request, name = self._queue.get()
                        if future == _CLOSE:
                            _LOGGER.debug("Closing thread")
                            self._queue.task_done()
                            return
            except Exception as e:
                _LOGGER.error(f"Error {e} occurred during {name} for {self._model_name}")
                future.set_exception(e)
                self._queue.task_done()
        _LOGGER.debug("Finishing worker thread")
