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
import socket
import time
from concurrent.futures import wait
from contextlib import closing
from typing import Callable

import numpy as np
import pytest

from pytriton.client import FuturesModelClient, ModelClient
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

_LOGGER = logging.getLogger(__name__)

_SMALL_TIMEOUT = 0.5
_LARGE_TIMEOUT = 1.5
_GARGANTUAN_TIMEOUT = 5.0


def find_free_ports(not_allowed_protocol=None):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as http_socket:
        http_socket.bind(("", 0))
        http_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as grpc_socket:
            grpc_socket.bind(("", 0))
            grpc_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as metrics_socket:
                metrics_socket.bind(("", 0))
                metrics_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                result = {}
                if not_allowed_protocol != "http":
                    result["http_port"] = http_socket.getsockname()[1]
                if not_allowed_protocol != "grpc":
                    result["grpc_port"] = grpc_socket.getsockname()[1]
                result["metrics_port"] = metrics_socket.getsockname()[1]
                return result


def triton_server_builder(ports):
    @batch
    def _infer_fn(**inputs):  # noqa: N803
        _LOGGER.debug("Inputs: %s", inputs)
        sleep_time = inputs["INPUT_1"].squeeze().item()
        _LOGGER.info("Will sleep %ss", sleep_time)
        time.sleep(sleep_time)
        return_value = {
            "OUTPUT_1": inputs["INPUT_1"],
        }
        _LOGGER.debug("Return value %s", return_value)
        return return_value

    class TritonInstance:
        """Context manager to hold Triton instance and ports"""

        # Type hints added for linter
        model_name: str
        infer_function: Callable
        config: TritonConfig

        def __init__(
            self,
            model_name,
            infer_function,
            grpc_port=None,
            http_port=None,
            metrics_port=None,
        ):
            self.grpc_port = grpc_port
            self.http_port = http_port
            self.metrics_port = metrics_port
            self.model_name = model_name
            kwargs = {}
            if self.grpc_port is not None:
                self.grpc_url = f"grpc://localhost:{self.grpc_port}"
            else:
                kwargs["allow_grpc"] = False
            if self.http_port is not None:
                self.http_url = f"http://localhost:{self.http_port}"
            else:
                kwargs["allow_http"] = False
            assert self.grpc_port is not None or self.http_port is not None
            self.config = TritonConfig(http_port=http_port, grpc_port=grpc_port, metrics_port=metrics_port, **kwargs)
            self.infer_function = infer_function

        def __enter__(self):
            self.triton = Triton(config=self.config)
            _LOGGER.debug("Binding %s model.", self.model_name)
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
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            _LOGGER.debug("Triton server stopping.")
            self.triton.stop()

        def get_model_client(self, init_timeout_s=_GARGANTUAN_TIMEOUT, inference_timeout_s=_GARGANTUAN_TIMEOUT):
            _LOGGER.debug(
                "Creating model client with init_timeout_s=%s and inference_timeout_s=%s",
                init_timeout_s,
                inference_timeout_s,
            )
            return ModelClient(
                self.http_url if self.http_port is not None else self.grpc_url,
                self.model_name,
                init_timeout_s=init_timeout_s,
                inference_timeout_s=inference_timeout_s,
            )

        def get_model_futures_client(self, init_timeout_s=_GARGANTUAN_TIMEOUT, inference_timeout_s=_GARGANTUAN_TIMEOUT):
            _LOGGER.debug(
                "Creating futures model client with init_timeout_s=%s and inference_timeout_s=%s",
                init_timeout_s,
                inference_timeout_s,
            )
            return FuturesModelClient(
                self.http_url if self.http_port is not None else self.grpc_url,
                self.model_name,
                init_timeout_s=init_timeout_s,
                inference_timeout_s=inference_timeout_s,
            )

    _LOGGER.debug("Using ports: %s", ports)
    with TritonInstance(
        model_name="Sleeper",
        infer_function=_infer_fn,
        **ports,
    ) as triton:
        yield triton


@pytest.fixture(
    params=[
        "http",
        "grpc",
        None,
    ],
    scope="function",
)
def non_allowed_protocol(request):
    return request.param


@pytest.fixture(
    params=[
        "ModelClient",
        "FuturesModelClient",
    ],
    scope="function",
)
def client_type(request):
    return request.param


# Define a fixture to create and return a Triton server instance
@pytest.fixture(scope="function")
def first_triton_server(non_allowed_protocol):
    _LOGGER.debug("Preparing first Triton server.")
    ports = find_free_ports(non_allowed_protocol)
    yield from triton_server_builder(ports)


# Define a fixture to create and return a Triton server instance
@pytest.fixture(scope="function")
def second_triton_server(non_allowed_protocol):
    _LOGGER.debug("Preparing second Triton server.")
    ports = find_free_ports(non_allowed_protocol)
    yield from triton_server_builder(ports)


@pytest.fixture(scope="function")
def first_client(first_triton_server, client_type):
    if client_type == "ModelClient":
        return first_triton_server.get_model_client()
    elif client_type == "FuturesModelClient":
        return first_triton_server.get_model_futures_client()
    else:
        raise ValueError(f"Unknown client type {client_type}")


@pytest.fixture(scope="function")
def second_client(second_triton_server, client_type):
    if client_type == "ModelClient":
        return second_triton_server.get_model_client()
    elif client_type == "FuturesModelClient":
        return second_triton_server.get_model_futures_client()
    else:
        raise ValueError(f"Unknown client type {client_type}")


@pytest.fixture(scope="session")
def input_sleep_large():
    _LOGGER.debug("Preparing input array with value %s.", _LARGE_TIMEOUT)
    yield np.array([[_LARGE_TIMEOUT]], dtype=np.float64)


def test_infer_sample_success_single(first_client, input_sleep_large, client_type):
    _LOGGER.debug("Testing async grpc_client with input %s", input_sleep_large)
    with first_client as first_client:
        if client_type == "ModelClient":
            first = first_client.infer_sample(input_sleep_large)
        elif client_type == "FuturesModelClient":
            first_future = first_client.infer_sample(input_sleep_large)
            done, _not_done = wait(
                [
                    first_future,
                ],
                timeout=_LARGE_TIMEOUT * 1.3,
            )
            assert len(done) == 1
            first = first_future.result()
        assert first["OUTPUT_1"] == input_sleep_large


def test_infer_sample_success_cohabit(first_client, second_client, input_sleep_large, client_type):
    _LOGGER.debug("Testing async grpc_client with input %s", input_sleep_large)
    with first_client as first_client:
        with second_client as second_client:
            if client_type == "ModelClient":
                first = first_client.infer_sample(input_sleep_large)
                second = second_client.infer_sample(input_sleep_large)
            elif client_type == "FuturesModelClient":
                first_future = first_client.infer_sample(input_sleep_large)
                second_future = second_client.infer_sample(input_sleep_large)
                done, _not_done = wait([first_future, second_future], timeout=_LARGE_TIMEOUT * 1.3)
                assert len(done) == 2
                first = first_future.result()
                second = second_future.result()
            assert first["OUTPUT_1"] == input_sleep_large
            assert second["OUTPUT_1"] == input_sleep_large
