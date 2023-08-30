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
"""Test of decoupled mode"""

import contextlib
import logging
import socket
import time

import numpy as np
import pytest

from pytriton.client import DecoupledModelClient, ModelClient
from pytriton.client.exceptions import (
    PyTritonClientInferenceServerError,
    PyTritonClientTimeoutError,
    PyTritonClientValueError,
)
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

_LOGGER = logging.getLogger(__name__)

_SMALL_TIMEOUT = 0.1
_TEST_TIMEOUT = 25.0
_GARGANTUAN_TIMEOUT = 10.0
_WRONG_TIMEOUT = -1.0
_CORRECT_REPEAT = 2
_WRONG_REPEAT = -2
_NO_REPEAT = 0


@pytest.fixture(scope="function")
def find_free_ports():
    """Fixture to find free ports for gprc, http, and metrics"""
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as grpc:
        grpc.bind(("", 0))
        grpc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as http:
            http.bind(("", 0))
            http.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as metrics:
                metrics.bind(("", 0))
                metrics.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                yield {
                    "grpc_port": grpc.getsockname()[1],
                    "http_port": http.getsockname()[1],
                    "metrics_port": metrics.getsockname()[1],
                }


class TritonInstance:

    """Context manager to hold Triton instance and ports"""

    def __init__(self, grpc_port, http_port, metrics_port, model_name, infer_function, decoupled=True):
        self.grpc_port = grpc_port
        self.http_port = http_port
        self.metrics_port = metrics_port
        self.model_name = model_name
        self.config = TritonConfig(http_port=http_port, grpc_port=grpc_port, metrics_port=metrics_port)
        self.infer_function = infer_function
        self.grpc_url = f"grpc://localhost:{self.grpc_port}"
        self.http_url = f"http://localhost:{self.http_port}"
        self.decoupled = decoupled

    def __enter__(self):
        try:
            _LOGGER.info("Checking if Triton server is already running.")
            if self.decoupled:
                DecoupledModelClient(
                    self.grpc_url,
                    self.model_name,
                    init_timeout_s=_SMALL_TIMEOUT,
                    inference_timeout_s=_SMALL_TIMEOUT,
                    lazy_init=False,
                )
            else:
                ModelClient(
                    self.grpc_url,
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
                Tensor(dtype=np.float64, shape=(-1,)),
                Tensor(dtype=np.int64, shape=(-1,)),
            ],
            outputs=[
                Tensor(dtype=np.float64, shape=(-1,)),
                Tensor(dtype=np.int64, shape=(-1,)),
            ],
            config=ModelConfig(decoupled=self.decoupled),
            strict=True,
        )
        _LOGGER.info("Running Triton server.")
        self.triton.run()
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


@pytest.fixture(scope="function")
def triton_decoupled_server(find_free_ports):
    @batch
    def _infer_fn(**inputs):  # noqa: N803
        _LOGGER.debug(f"Inputs: {inputs}")
        sleep_time = inputs["INPUT_1"].squeeze().item()
        response_counter = inputs["INPUT_2"].squeeze().item()
        if sleep_time < 0:
            raise ValueError("Sleep time must be positive.")
        if response_counter < 0:
            response_counter = -response_counter
            _LOGGER.info(f"Sleeper will raise ValueError after {response_counter} responses.")
            for _ in range(response_counter):
                _LOGGER.info(f"Will sleep {sleep_time}s")
                time.sleep(sleep_time)
                return_value = {
                    "OUTPUT_1": inputs["INPUT_1"],
                    "OUTPUT_2": inputs["INPUT_2"],
                }
                _LOGGER.debug(f"Yield value {return_value}")
                yield return_value
            time.sleep(sleep_time)
            _LOGGER.info(f"Will sleep {sleep_time}s")
            raise ValueError("Response counter must be positive.")
        else:
            _LOGGER.info(f"Sleeper will succed after {response_counter} responses.")
            for _ in range(response_counter):
                _LOGGER.info(f"Will sleep {sleep_time}s")
                time.sleep(sleep_time)
                return_value = {
                    "OUTPUT_1": inputs["INPUT_1"],
                    "OUTPUT_2": inputs["INPUT_2"],
                }
                _LOGGER.debug(f"Yield value {return_value}")
                yield return_value

    _LOGGER.debug(f"Using ports: grpc={find_free_ports}")
    with TritonInstance(**find_free_ports, model_name="Sleeper", infer_function=_infer_fn, decoupled=True) as triton:
        yield triton


@pytest.fixture(scope="function")
def triton_coupled_server(find_free_ports):
    @batch
    def _infer_fn(**inputs):  # noqa: N803
        _LOGGER.debug(f"Inputs: {inputs}")
        sleep_time = inputs["INPUT_1"].squeeze().item()
        if sleep_time < 0:
            raise ValueError("Sleep time must be positive.")
        _LOGGER.info(f"Will sleep {sleep_time}s")
        time.sleep(sleep_time)
        return_value = {
            "OUTPUT_1": inputs["INPUT_1"],
            "OUTPUT_2": inputs["INPUT_2"],
        }
        _LOGGER.debug(f"Yield value {return_value}")
        return return_value

    _LOGGER.debug(f"Using ports: grpc={find_free_ports}")
    with TritonInstance(**find_free_ports, model_name="Sleeper", infer_function=_infer_fn, decoupled=False) as triton:
        yield triton


@pytest.fixture(scope="function")
def grpc_decoupled_client_server(triton_decoupled_server):
    _LOGGER.debug(
        f"Preparing client for {triton_decoupled_server.grpc_url} with init timeout {_GARGANTUAN_TIMEOUT} and inference timeout {_SMALL_TIMEOUT}."
    )
    yield DecoupledModelClient(
        url=triton_decoupled_server.grpc_url,
        model_name=triton_decoupled_server.model_name,
        init_timeout_s=_GARGANTUAN_TIMEOUT,
        inference_timeout_s=_SMALL_TIMEOUT,
    )


@pytest.fixture(scope="function")
def grpc_decoupled_client_coupled_server(triton_coupled_server):
    _LOGGER.debug(
        f"Preparing client for {triton_coupled_server.grpc_url} with init timeout {_GARGANTUAN_TIMEOUT} and inference timeout {_SMALL_TIMEOUT}."
    )
    yield DecoupledModelClient(
        url=triton_coupled_server.grpc_url,
        model_name=triton_coupled_server.model_name,
        init_timeout_s=_GARGANTUAN_TIMEOUT,
        inference_timeout_s=_SMALL_TIMEOUT,
    )


@pytest.fixture(scope="function")
def grpc_coupled_client_decoupled_server(triton_decoupled_server):
    _LOGGER.debug(
        f"Preparing client for {triton_decoupled_server.grpc_url} with init timeout {_GARGANTUAN_TIMEOUT} and inference timeout {_SMALL_TIMEOUT}."
    )
    yield ModelClient(
        url=triton_decoupled_server.grpc_url,
        model_name=triton_decoupled_server.model_name,
        init_timeout_s=_GARGANTUAN_TIMEOUT,
        inference_timeout_s=_SMALL_TIMEOUT,
    )


@pytest.fixture(scope="function")
def http_coupled_client_decoupled_server(triton_decoupled_server):
    _LOGGER.debug(
        f"Preparing client for {triton_decoupled_server.grpc_url} with init timeout {_GARGANTUAN_TIMEOUT} and inference timeout {_SMALL_TIMEOUT}."
    )
    yield ModelClient(
        url=triton_decoupled_server.http_url,
        model_name=triton_decoupled_server.model_name,
        init_timeout_s=_GARGANTUAN_TIMEOUT,
        inference_timeout_s=_SMALL_TIMEOUT,
    )


@pytest.mark.timeout(_TEST_TIMEOUT)
def test_coupled_infer_sample_failure_http(grpc_coupled_client_decoupled_server):
    with pytest.raises(PyTritonClientInferenceServerError):
        with grpc_coupled_client_decoupled_server as client:
            client.infer_sample(np.array([_SMALL_TIMEOUT]), np.array([1]))


@pytest.mark.timeout(_TEST_TIMEOUT)
def test_coupled_infer_batch_failure_http(grpc_coupled_client_decoupled_server):
    with pytest.raises(PyTritonClientInferenceServerError):
        with grpc_coupled_client_decoupled_server as client:
            client.infer_batch(np.array([[_SMALL_TIMEOUT]]), np.array([[1]]))


@pytest.mark.timeout(_TEST_TIMEOUT)
def test_decoupled_init_failure_http(triton_decoupled_server):
    with pytest.raises(PyTritonClientValueError):
        DecoupledModelClient(
            url=triton_decoupled_server.http_url,
            model_name=triton_decoupled_server.model_name,
        )


@pytest.mark.timeout(_TEST_TIMEOUT)
def test_decoupled_infer_sample_success_grpc(grpc_decoupled_client_server):
    with grpc_decoupled_client_server as client:
        responses = list(client.infer_sample(np.array([_SMALL_TIMEOUT]), np.array([_CORRECT_REPEAT])))
        assert len(responses) == 2
        assert responses[0]["OUTPUT_1"] == _SMALL_TIMEOUT
        assert responses[0]["OUTPUT_2"] == _CORRECT_REPEAT
        assert responses[1]["OUTPUT_1"] == _SMALL_TIMEOUT
        assert responses[1]["OUTPUT_2"] == _CORRECT_REPEAT


@pytest.mark.timeout(_TEST_TIMEOUT)
def test_decoupled_infer_batch_success_grpc(grpc_decoupled_client_server):
    with grpc_decoupled_client_server as client:
        responses = list(client.infer_batch(np.array([[_SMALL_TIMEOUT]]), np.array([[_CORRECT_REPEAT]])))
        assert len(responses) == 2
        assert responses[0]["OUTPUT_1"] == _SMALL_TIMEOUT
        assert responses[0]["OUTPUT_2"] == _CORRECT_REPEAT
        assert responses[1]["OUTPUT_1"] == _SMALL_TIMEOUT
        assert responses[1]["OUTPUT_2"] == _CORRECT_REPEAT


@pytest.mark.timeout(_TEST_TIMEOUT)
def test_decoupled_infer_sample_failure_coupled_server_grpc(grpc_decoupled_client_coupled_server):
    with grpc_decoupled_client_coupled_server as client:
        with pytest.raises(PyTritonClientInferenceServerError):
            client.infer_sample(np.array([_SMALL_TIMEOUT]), np.array([_CORRECT_REPEAT]))


@pytest.mark.timeout(_TEST_TIMEOUT)
def test_decoupled_infer_batch_failure_coupled_server_grpc(grpc_decoupled_client_coupled_server):
    with grpc_decoupled_client_coupled_server as client:
        with pytest.raises(PyTritonClientInferenceServerError):
            client.infer_batch(np.array([[_SMALL_TIMEOUT]]), np.array([[_CORRECT_REPEAT]]))


@pytest.mark.timeout(_TEST_TIMEOUT)
def test_decoupled_infer_sample_fast_failure_no_iter_grpc(grpc_decoupled_client_server):
    with grpc_decoupled_client_server as client:
        client.infer_sample(np.array([_WRONG_TIMEOUT]), np.array([_CORRECT_REPEAT]))


@pytest.mark.timeout(_TEST_TIMEOUT)
def test_decoupled_infer_sample_fast_failure_iter_grpc(grpc_decoupled_client_server):
    with grpc_decoupled_client_server as client:
        with pytest.raises(PyTritonClientInferenceServerError):
            list(client.infer_sample(np.array([_WRONG_TIMEOUT]), np.array([_CORRECT_REPEAT])))


@pytest.mark.timeout(_TEST_TIMEOUT)
def test_decoupled_infer_sample_slow_failure_iter_grpc(grpc_decoupled_client_server):
    with grpc_decoupled_client_server as client:
        iterator = client.infer_sample(np.array([_SMALL_TIMEOUT]), np.array([_WRONG_REPEAT]))
        first_result = next(iterator)
        assert first_result["OUTPUT_1"] == _SMALL_TIMEOUT
        assert first_result["OUTPUT_2"] == _WRONG_REPEAT
        second_result = next(iterator)
        assert second_result["OUTPUT_1"] == _SMALL_TIMEOUT
        assert second_result["OUTPUT_2"] == _WRONG_REPEAT
        with pytest.raises(PyTritonClientInferenceServerError):
            next(iterator)
