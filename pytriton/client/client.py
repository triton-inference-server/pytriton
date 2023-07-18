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

import itertools
import logging
import socket
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tritonclient.grpc
import tritonclient.http
import tritonclient.utils

from pytriton.client.exceptions import (
    PyTritonClientClosedError,
    PyTritonClientInferenceServerError,
    PyTritonClientModelDoesntSupportBatchingError,
    PyTritonClientTimeoutError,
    PyTritonClientValueError,
)
from pytriton.client.utils import (
    _DEFAULT_NETWORK_TIMEOUT_S,
    _DEFAULT_WAIT_FOR_MODEL_TIMEOUT_S,
    create_client_from_url,
    get_client_lib_from_client,
    get_model_config,
    wait_for_model_ready,
)
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


class ModelClient:
    """Synchronous client for interacting with a model deployed on the Triton Inference Server.

    This client provides a synchronous way to interact with a Triton Inference Server model. It can be used to perform
    inference on a model by providing input data and receiving the corresponding output data. The client can be used in
    a `with` statement to ensure proper resource management.

    Example usage:

        ```python
        with ModelClient("localhost", "MyModel") as client:
            result_dict = client.infer_batch(input1=input1_data, input2=input2_data)
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

        self._general_client = create_client_from_url(url, network_timeout_s=self._network_timeout_s)
        self._infer_client = create_client_from_url(url, network_timeout_s=self._inference_timeout_s)

        self._triton_client_lib = get_client_lib_from_client(self._general_client)

        self._model_name = model_name
        self._model_version = model_version

        self._request_id_generator = itertools.count(0)
        self._model_config = None
        self._model_ready = None
        self._lazy_init = lazy_init

        self._infer_extra_kwargs = (
            {"client_timeout": self._inference_timeout_s}
            if isinstance(self._infer_client, tritonclient.grpc.InferenceServerClient)
            else {}
        )

        if not self._lazy_init:
            self._wait_and_init_model_config(self._init_timeout_s)

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
                request_id=str(next(self._request_id_generator)),
                parameters=parameters,
                **self._infer_extra_kwargs,
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
            raise PyTritonClientTimeoutError(
                f"Timeout occurred during inference request. Timeout: {self._inference_timeout_s} s. Message: {e}"
            ) from e

        if isinstance(response, tritonclient.http.InferResult):
            outputs = {
                output["name"]: response.as_numpy(output["name"]) for output in response.get_response()["outputs"]
            }
        else:
            outputs = {output.name: response.as_numpy(output.name) for output in response.get_response().outputs}

        return outputs


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
        self._thread_pool_executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._thread_clients = {}
        self._init_timeout_s = _DEFAULT_FUTURES_INIT_TIMEOUT_S if init_timeout_s is None else init_timeout_s
        self._inference_timeout_s = inference_timeout_s

    def __enter__(self):
        """Create context for using FuturesModelClient as a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close resources used by FuturesModelClient instance when exiting from the context."""
        self.close()

    def close(self, wait=True, *, cancel_futures=False):
        """Close resources used by FuturesModelClient.

        This method closes the resources used by the FuturesModelClient instance, including the Triton Inference Server connections.
        Once this method is called, the FuturesModelClient instance should not be used again.

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
        try:
            return self._thread_pool_executor.submit(lambda: self._get_client(lazy_init=True).wait_for_model(timeout_s))
        except RuntimeError as e:
            raise PyTritonClientClosedError("FutureModelClient is already closed") from e

    def model_config(self) -> Future:
        """Obtain the configuration of the model deployed on the Triton Inference Server.

        This method returns a Future object that will contain the TritonModelConfig object when it is ready.
        Client will wait init_timeout_s for the server to get into readiness state before obtaining the model configuration.

        Returns:
            A Future object that will contain the TritonModelConfig object when it is ready.

        Raises:
            PyTritonClientClosedError: If the FuturesModelClient is closed.
        """
        try:
            return self._thread_pool_executor.submit(lambda: self._get_client().model_config)
        except RuntimeError as e:
            raise PyTritonClientClosedError("FutureModelClient is already closed") from e

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
        try:
            return self._thread_pool_executor.submit(
                lambda: self._get_client().infer_sample(
                    *inputs, parameters=parameters, headers=headers, **named_inputs
                ),
            )
        except RuntimeError as e:
            raise PyTritonClientClosedError("FutureModelClient is already closed") from e

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
        try:
            return self._thread_pool_executor.submit(
                lambda: self._get_client().infer_batch(*inputs, parameters=parameters, headers=headers, **named_inputs),
            )
        except RuntimeError as e:
            raise PyTritonClientClosedError("FutureModelClient is already closed") from e

    def _get_client(self, lazy_init: bool = False):
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
