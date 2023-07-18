#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
"""Test of network timeouts with pytest"""

import logging
import time

import numpy as np
import pytest

from pytriton.client import ModelClient
from pytriton.client.exceptions import PyTritonClientInferenceServerError, PyTritonClientTimeoutError
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
from tests.utils import find_free_port

_LOGGER = logging.getLogger(__name__)

_SMALLEST_TIMEOUT = 0.0
_SMALL_TIMEOUT = 0.5
_LARGE_TIMEOUT = 1.5
_TEST_TIMEOUT = 25.0
_GARGANTUAN_TIMEOUT = 10.0
_WRONG_TIMEOUT = -1.0


# Define a fixture to start and stop the Triton server with the Sleeper model
@pytest.fixture(scope="function")
def triton_server():
    @batch
    def _infer_fn(**inputs):  # noqa: N803
        _LOGGER.debug(f"Inputs: {inputs}")
        sleep_time = inputs["INPUT_1"].squeeze().item()
        _LOGGER.info(f"Will sleep {sleep_time}s")
        time.sleep(sleep_time)
        return_value = {
            "OUTPUT_1": inputs["INPUT_1"],
        }
        _LOGGER.debug(f"Return value {return_value}")
        return return_value

    class TritonInstance:

        """Context manager to hold Triton instance and ports"""

        def __init__(self, grpc_port, http_port, model_name, infer_function):
            self.grpc_port = grpc_port
            self.http_port = http_port
            self.model_name = model_name
            self.config = TritonConfig(http_port=http_port, grpc_port=grpc_port)
            self.infer_function = infer_function
            self.grpc_url = f"grpc://localhost:{self.grpc_port}"
            self.http_url = f"http://localhost:{self.http_port}"

        def __enter__(self):
            try:
                _LOGGER.info("Checking if Triton server is already running.")
                ModelClient(
                    self.http_url,
                    self.model_name,
                    init_timeout_s=_SMALL_TIMEOUT,
                    inference_timeout_s=_SMALL_TIMEOUT,
                    lazy_init=False,
                )
                message = "Triton server already running."
                _LOGGER.error(message)
                raise RuntimeError(message)
            except PyTritonClientTimeoutError:
                _LOGGER.debug("Triton server not running.")
                pass
            self.triton = Triton(config=self.config)
            _LOGGER.debug(f"Binding {self.model_name} model.")
            self.triton.bind(
                model_name=self.model_name,
                infer_func=self.infer_function,
                inputs=[
                    Tensor(dtype=np.float64, shape=(-1, 1)),
                ],
                outputs=[
                    Tensor(dtype=np.float64, shape=(-1, 1)),
                ],
                config=ModelConfig(max_batch_size=128),
            )
            _LOGGER.info("Running Triton server.")
            self.triton.run()
            _LOGGER.debug("Waiting for Triton server to load model.")
            with ModelClient(
                self.http_url,
                self.model_name,
                init_timeout_s=_GARGANTUAN_TIMEOUT,
                inference_timeout_s=_GARGANTUAN_TIMEOUT,
                lazy_init=False,
            ) as client:
                _LOGGER.info(f"Triton server ready. {client.model_config}")
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            _LOGGER.debug("Triton server stopping.")
            self.triton.stop()
            _LOGGER.debug("Checking if Triton server is still running.")
            while True:
                try:
                    with ModelClient(
                        self.http_url,
                        self.model_name,
                        init_timeout_s=_SMALL_TIMEOUT,
                        inference_timeout_s=_SMALL_TIMEOUT,
                        lazy_init=False,
                    ) as client:
                        _LOGGER.info(f"Triton server still running. {client.model_config}")
                except PyTritonClientTimeoutError:
                    _LOGGER.debug("Triton server not running.")
                    break
                _LOGGER.debug(f"Triton server still alive, so sleeping for {_SMALL_TIMEOUT}s.")
                time.sleep(_SMALL_TIMEOUT)
            _LOGGER.info("Triton server stopped.")

    grpc_port = find_free_port()
    http_port = find_free_port()
    _LOGGER.debug(f"Using ports: grpc={grpc_port}, http={http_port}")
    with TritonInstance(
        grpc_port=grpc_port, http_port=http_port, model_name="Sleeper", infer_function=_infer_fn
    ) as triton:
        yield triton


# Define a fixture to create and return a client object with a very small timeout value
@pytest.fixture(scope="function")
def http_client(triton_server):
    _LOGGER.debug(
        f"Preparing client for {triton_server.http_url} with init timeout {_GARGANTUAN_TIMEOUT} and inference timeout {_SMALL_TIMEOUT}."
    )
    return ModelClient(
        url=triton_server.http_url,
        model_name=triton_server.model_name,
        init_timeout_s=_GARGANTUAN_TIMEOUT,
        inference_timeout_s=_SMALL_TIMEOUT,
    )


# Define a fixture to create and return a client object with a very small timeout value
@pytest.fixture(scope="function")
def grpc_client(triton_server):
    _LOGGER.debug(
        f"Preparing client for {triton_server.grpc_url} with init timeout {_GARGANTUAN_TIMEOUT} and inference timeout {_SMALL_TIMEOUT}."
    )
    return ModelClient(
        url=triton_server.grpc_url,
        model_name=triton_server.model_name,
        init_timeout_s=_GARGANTUAN_TIMEOUT,
        inference_timeout_s=_SMALL_TIMEOUT,
    )


# Define a fixture to create and return an input array with a value of 1.5 seconds
@pytest.fixture(scope="session")
def input_sleep_large():
    _LOGGER.debug(f"Preparing input array with value {_LARGE_TIMEOUT}.")
    return np.array([[_LARGE_TIMEOUT]], dtype=np.float64)


# Define a fixture to create and return an input array with a value of -1 seconds
@pytest.fixture(scope="session")
def input_sleep_wrong():
    _LOGGER.debug(f"Preparing input array with value {_LARGE_TIMEOUT}.")
    return np.array([[_WRONG_TIMEOUT]], dtype=np.float64)


# Define a fixture to create and return an input array with a value of 1.5 seconds
@pytest.fixture(scope="session")
def input_sleep_smallest():
    _LOGGER.debug(f"Preparing input array with value {_LARGE_TIMEOUT}.")
    return np.array([[_SMALLEST_TIMEOUT]], dtype=np.float64)


@pytest.mark.timeout(_TEST_TIMEOUT)
def test_infer_sample_network_timeout_grpc(triton_server, grpc_client, input_sleep_large):
    _LOGGER.debug(f"Testing grpc_client with input {input_sleep_large}.")
    with pytest.raises(PyTritonClientTimeoutError):
        grpc_client.infer_sample(input_sleep_large)


@pytest.mark.timeout(_TEST_TIMEOUT)
def test_infer_sample_network_timeout_http(triton_server, http_client, input_sleep_large):
    _LOGGER.debug(f"Testing http_client with input {input_sleep_large}.")
    with pytest.raises(PyTritonClientTimeoutError):
        http_client.infer_sample(input_sleep_large)


@pytest.mark.timeout(_TEST_TIMEOUT)
def test_infer_sample_model_failure_grpc(triton_server, grpc_client, input_sleep_wrong):
    _LOGGER.debug(f"Testing grpc_client with input {input_sleep_wrong}.")
    with pytest.raises(PyTritonClientInferenceServerError):
        grpc_client.infer_sample(input_sleep_wrong)


@pytest.mark.timeout(_TEST_TIMEOUT)
def test_infer_sample_model_failure_http(triton_server, http_client, input_sleep_wrong):
    _LOGGER.debug(f"Testing http_client with input {input_sleep_wrong}.")
    with pytest.raises(PyTritonClientInferenceServerError):
        http_client.infer_sample(input_sleep_wrong)


@pytest.mark.timeout(_TEST_TIMEOUT)
def test_infer_sample_success_grpc(triton_server, grpc_client, input_sleep_smallest):
    _LOGGER.debug(f"Testing grpc_client with input {input_sleep_smallest}.")
    result = grpc_client.infer_sample(input_sleep_smallest)
    assert result["OUTPUT_1"] == input_sleep_smallest


@pytest.mark.timeout(_TEST_TIMEOUT)
def test_infer_sample_success_http(triton_server, http_client, input_sleep_smallest):
    _LOGGER.debug(f"Testing http_client with input {input_sleep_smallest}.")
    result = http_client.infer_sample(input_sleep_smallest)
    assert result["OUTPUT_1"] == input_sleep_smallest
