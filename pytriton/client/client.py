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
import socket
import sys
import threading
import time
import urllib.parse
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Optional, Tuple, Union

import gevent
import numpy as np
import tritonclient.grpc
import tritonclient.http
import tritonclient.utils

from pytriton.client.exceptions import (
    PyTritonClientInferenceServerError,
    PyTritonClientModelDoesntSupportBatchingError,
    PyTritonClientTimeoutError,
    PyTritonClientUrlParseError,
    PyTritonClientValueError,
)
from pytriton.client.utils import _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S, get_model_config, wait_for_model_ready
from pytriton.constants import DEFAULT_GRPC_PORT, DEFAULT_HTTP_PORT

_LOGGER = logging.getLogger(__name__)

_DEFAULT_INIT_TIMEOUT_S = _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S
_DEFAULT_INFERENCE_TIMEOUT_S = 60.0
_DEFAULT_ASYNC_INIT_TIMEOUT_S = 30.0

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
        inference_timeout_s: Optional[float] = _DEFAULT_INFERENCE_TIMEOUT_S,
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
        if not isinstance(url, str):
            raise PyTritonClientUrlParseError(f"Could not parse url {url}")

        parsed_url = urllib.parse.urlparse(url)
        if not parsed_url.scheme or parsed_url.scheme.lower() not in ["grpc", "http"]:
            _LOGGER.debug(f"Adding http scheme to {url}")
            parsed_url = urllib.parse.urlparse(f"http://{url}")
        self._scheme = parsed_url.scheme.lower()

        port = parsed_url.port or {"grpc": DEFAULT_GRPC_PORT, "http": DEFAULT_HTTP_PORT}[parsed_url.scheme.lower()]
        self._url = f"{parsed_url.hostname}:{port}"
        self._model_name = model_name
        self._model_version = model_version

        self._triton_client_lib = {"grpc": tritonclient.grpc, "http": tritonclient.http}[parsed_url.scheme.lower()]
        _LOGGER.debug(f"Creating InferenceServerClient for {parsed_url.scheme}://{self._url}")

        # Monkey patch __del__ method from client to catch error in client when instance is garbage collected.
        # This is needed because we are closing client in __exit__ method or in close method.
        # (InferenceClient uses gevent library which does not support closing twice from different threads)
        self._monkey_patch_client()

        self._init_timeout_s = init_timeout_s
        self._inference_timeout_s = inference_timeout_s

        kwargs = self._get_init_extra_args()
        self._client = self._triton_client_lib.InferenceServerClient(self._url, **kwargs)

        self._request_id_generator = itertools.count(0)
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
        wait_for_model_ready(
            self._client,
            self._model_name,
            self._model_version,
            init_timeout_s=timeout_s,
            # Tritonclient doesn't support timeout yet
            # grpc_network_timeout_s=self._inference_timeout_s,
        )

    @property
    def is_batching_supported(self):
        """Checks if model supports batching.

        Also waits for server to get into readiness state.
        """
        return self.model_config.max_batch_size > 0

    @property
    def model_config(self):
        """Obtain configuration of model deployed on the Triton Inference Server.

        Also waits for server to get into readiness state.
        """
        if not self._model_config:
            kwargs = self._get_model_config_extra_args()
            self._model_config = get_model_config(
                self._client,
                self._model_name,
                self._model_version,
                init_timeout_s=self._init_timeout_s,
                **kwargs,
            )
        return self._model_config

    def infer_sample(
        self,
        *inputs,
        parameters: Optional[Dict[str, Union[str, int, bool]]] = None,
        headers: Optional[Dict[str, Union[str, int, bool]]] = None,
        **named_inputs,
    ) -> Dict[str, np.ndarray]:
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
            parameters: custom inference parameters.
            headers: custom inference headers.
            **named_inputs: inference inputs provided as named arguments.

        Returns:
            dictionary with inference results, where dictionary keys are output names.

        Raises:
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

            with ModelClient("localhost", "MyModel") as client:
                result_dict = client.infer_sample(input1, input2)

        Inference inputs can be provided either as positional or keyword arguments:

            result_dict = client.infer_batch(input1, input2)
            result_dict = client.infer_batch(a=input1, b=input2)

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
                and wait time for server and model being ready exceeds `init_timeout_s` or
                inference time exceeds `inference_timeout_s` passed to `__init__`.
            PyTritonClientModelUnavailableError: If model with given name (and version) is unavailable.
            PyTritonClientInferenceServerError:
                If error occurred on inference callable or Triton Inference Server side,
        """
        _verify_inputs_args(inputs, named_inputs)
        _verify_parameters(parameters)
        _verify_parameters(headers)

        if not self.is_batching_supported:
            raise PyTritonClientModelDoesntSupportBatchingError(
                f"Model {self.model_config.model_name} doesn't support batching - use infer_sample method instead"
            )

        return self._infer(inputs or named_inputs, parameters, headers)

    def _monkey_patch_client(self):
        """Monkey patch InferenceServerClient to catch error in __del__."""
        if not hasattr(self._triton_client_lib.InferenceServerClient, "__del__"):
            return

        old_del = self._triton_client_lib.InferenceServerClient.__del__

        def _monkey_patched_del(self):
            """Monkey patched del."""
            try:
                old_del(self)
            except gevent.exceptions.InvalidThreadUseError:
                _LOGGER.warning("gevent.exceptions.InvalidThreadUseError in __del__ of InferenceServerClient")
            except Exception as e:
                _LOGGER.error("Exception in __del__ of InferenceServerClient: %s", e)

        self._triton_client_lib.InferenceServerClient.__del__ = _monkey_patched_del

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
        kwargs = self._get_model_config_extra_args()
        self._model_config = get_model_config(
            self._client,
            self._model_name,
            self._model_version,
            init_timeout_s=timeout_s,
            **kwargs,
        )

    def _infer(self, inputs: _IOType, parameters, headers) -> Dict[str, np.ndarray]:

        if self._model_ready:
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
            kwargs = self._get_infer_extra_args()
            _LOGGER.debug("Sending inference request to Triton Inference Server")
            response = self._client.infer(
                model_name=self._model_name,
                model_version=self._model_version or "",
                inputs=inputs_wrapped,
                headers=headers,
                outputs=outputs_wrapped,
                request_id=str(next(self._request_id_generator)),
                parameters=parameters,
                **kwargs,
            )
        except tritonclient.utils.InferenceServerException as e:
            if (
                "Deadline Exceeded" in e.message()
            ):  # tritonclient.grpc raises execption with message Deadline Exceeded for timeout
                message = f"Timeout occurred during inference request. Timeout: {self._inference_timeout_s} s Message: {e.message()}"
                _LOGGER.error(message)
                raise PyTritonClientTimeoutError(message) from e
            else:  # both clients raise tritonclient.utils.InferenceServerException for erros at server side
                message = f"Error occurred during inference request. Message: {e.message()}"
                _LOGGER.error(message)
                raise PyTritonClientInferenceServerError(message) from e
        except socket.timeout as e:  # tritonclient.http raises socket.timeout for timeout
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

    def _get_init_extra_args(self):
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
            "network_timeout": self._inference_timeout_s,
            # This value sets the maximum time allowed for establishing a connection to the server.
            # We use the inference timeout here instead of the init timeout because the init timeout
            # is meant for waiting for the model to be ready. The connection timeout should be shorter
            # than the init timeout because it only checks if connection is established (e.g. correct port)
            "connection_timeout": self._inference_timeout_s,
        }
        return kwargs

    def _get_model_config_extra_args(self):
        # For the GRPC protocol, the timeout must be passed to the each request as client_timeout
        # model_config doesn't yet support timeout but it is planned for the future
        # grpc_network_timeout_s will be used for model_config
        return {}


class FuturesModelClient:
    """A client for model deployed on the Triton Inference Server using concurrent.futures.

    This client allows asynchronous inference requests using a thread pool executor.

    Example:
    ```python
    with FuturesModelClient("localhost", "BERT") as client
         result_future = client.infer_sample(input1_sample, input2_sample)
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
        init_timeout_s: float = _DEFAULT_ASYNC_INIT_TIMEOUT_S,
        inference_timeout_s: Optional[float] = _DEFAULT_INFERENCE_TIMEOUT_S,
    ):
        """Initializes the FuturesModelClient for a given model.

        Args:
            url (str): The Triton Inference Server url, e.g. ```grpc://localhost:8001```.
            model_name (str): The name of the model to interact with.
            model_version (str or None): The version of the model to interact with.
                If None, the latest version will be used.
            max_workers (int or None): The maximum number of threads that can be used to execute
                the given calls. If None, the default value will be used.
            init_timeout_s (float): The timeout for server and model being ready.
                If None, the default value will be used.
            inference_timeout_s: timeout in seconds for the model inference process.
                If non passed default 60 seconds timeout will be used.
                For HTTP client it is not only inference timeout but any client request timeout
                - get model config, is model loaded. For GRPC client it is only inference timeout.
        """
        self._url = url
        self._model_name = model_name
        self._model_version = model_version
        self._thread_pool_executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._thread_clients = {}
        self._init_timeout_s = init_timeout_s
        self._inference_timeout_s = inference_timeout_s

    def __enter__(self):
        """Create context for use FuturesModelClient as a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close resources used by FuturesModelClient when exiting from context."""
        self.close()

    def close(self, wait=True, *, cancel_futures=False):
        """Close resources used by FuturesModelClient.

        Args:
            wait: If True, then shutdown will not return until all running futures have finished executing.
            cancel_futures: If True, then all pending futures that have not yet started executing will be cancelled.
                If False, then pending futures are left in the queue and will be run if not cancelled before they
                expire. This argument is ignored for Python < 3.9.
        """
        _LOGGER.debug("Closing FuturesModelClient.")
        if sys.version_info >= (3, 9) and cancel_futures:
            # Cancel futures argument was introduced from Python version 3.9
            self._thread_pool_executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        else:
            if cancel_futures:
                # Log a warning message that cancel_futures is not supported for older Python
                _LOGGER.warning("cancel_futures argument is ignored for Python < 3.9")
            self._thread_pool_executor.shutdown(wait=wait)
        with self._lock:
            for client in self._thread_clients.values():
                client.close()

    def wait_for_model(self, timeout_s: float = _DEFAULT_ASYNC_INIT_TIMEOUT_S) -> Future:
        """Returns future, which is set to True, when model is ready.

        Args:
            timeout_s: timeout to server and model get into readiness state.

        """
        return self._thread_pool_executor.submit(lambda: self._get_client(lazy_init=True).wait_for_model(timeout_s))

    def model_config(self):
        """Obtain configuration of model deployed on the Triton Inference Server.

        Also creates future, which waits for server to get into readiness state.
        """
        return self._thread_pool_executor.submit(lambda: self._get_client().model_config)

    def infer_sample(
        self,
        *inputs,
        parameters: Optional[Dict[str, Union[str, int, bool]]] = None,
        headers: Optional[Dict[str, Union[str, int, bool]]] = None,
        **named_inputs,
    ) -> Future:
        """Run asynchronous inference on single data sample and return a Future object.

        Typical usage:
        ````python
        with FuturesModelClient("localhost", "BERT") as client
            future = client.infer_sample(input1_sample, input2_sample)
            # do something else
            print(future.result())
        ````

        Inference inputs can be provided either as positional or keyword arguments:
        ```python
        future = client.infer_sample(input1, input2)
        future = client.infer_sample(a=input1, b=input2)
        ```

        Mixing of argument passing conventions is not supported and will raise PyTritonClientRuntimeError.

        Args:
            *inputs: inference inputs provided as positional arguments.
            parameters: optional dictionary of inference parameters.
            headers: optional dictionary of HTTP headers for the inference request.
            **named_inputs: inference inputs provided as named arguments.

        Returns:
            Future object wrapping a dictionary of inference results, where dictionary keys are output names.

        """
        # return a Future object that wraps the ModelClient infer_sample method
        return self._thread_pool_executor.submit(
            lambda: self._get_client().infer_sample(*inputs, parameters=parameters, headers=headers, **named_inputs),
        )

    def infer_batch(
        self,
        *inputs,
        parameters: Optional[Dict[str, Union[str, int, bool]]] = None,
        headers: Optional[Dict[str, Union[str, int, bool]]] = None,
        **named_inputs,
    ) -> Future:
        """Run asynchronous inference on batched data and return a Future object.

        Typical usage:
        ```python
        with FuturesModelClient("localhost", "BERT") as client
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
            *inputs: inference inputs provided as positional arguments.
            parameters: optional dictionary of inference parameters.
            headers: optional dictionary of HTTP headers for the inference request.
            **named_inputs: inference inputs provided as named arguments.

        Returns:
            Future object wrapping a dictionary of inference results, where dictionary keys are output names.

        """
        # return a Future object that wraps the ModelClient infer_batch method
        return self._thread_pool_executor.submit(
            lambda: self._get_client().infer_batch(*inputs, parameters=parameters, headers=headers, **named_inputs),
        )

    def _get_client(self, lazy_init: bool = False):
        """Get client from pool or create new one if pool is empty."""
        thread_id = threading.get_ident()
        with self._lock:
            if thread_id not in self._thread_clients:
                client = ModelClient(
                    self._url,
                    self._model_name,
                    self._model_version,
                    lazy_init=lazy_init,
                    init_timeout_s=self._init_timeout_s,
                    inference_timeout_s=self._inference_timeout_s,
                )
                self._thread_clients[thread_id] = client
            else:
                client = self._thread_clients[thread_id]
        return client
