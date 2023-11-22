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


def find_free_ports():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s2:
            s2.bind(("", 0))
            s2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s3:
                s3.bind(("", 0))
                s3.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                return {
                    "http_port": s.getsockname()[1],
                    "grpc_port": s2.getsockname()[1],
                    "metrics_port": s3.getsockname()[1],
                }


def triton_server_builder(ports):
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

        def __init__(self, grpc_port, http_port, metrics_port, model_name, infer_function):
            self.grpc_port = grpc_port
            self.http_port = http_port
            self.metrics_port = metrics_port
            self.model_name = model_name
            self.config = TritonConfig(http_port=http_port, grpc_port=grpc_port, metrics_port=metrics_port)
            self.infer_function = infer_function
            self.grpc_url = f"grpc://localhost:{self.grpc_port}"
            self.http_url = f"http://localhost:{self.http_port}"

        def __enter__(self):
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
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            _LOGGER.debug("Triton server stopping.")
            self.triton.stop()

        def get_model_client(self, init_timeout_s=_GARGANTUAN_TIMEOUT, inference_timeout_s=_GARGANTUAN_TIMEOUT):
            _LOGGER.debug(
                f"Creating model client with init_timeout_s={init_timeout_s} and inference_timeout_s={inference_timeout_s}"
            )
            return ModelClient(
                self.http_url,
                self.model_name,
                init_timeout_s=init_timeout_s,
                inference_timeout_s=inference_timeout_s,
            )

        def get_model_futures_client(self, init_timeout_s=_GARGANTUAN_TIMEOUT, inference_timeout_s=_GARGANTUAN_TIMEOUT):
            _LOGGER.debug(
                f"Creating futures model client with init_timeout_s={init_timeout_s} and inference_timeout_s={inference_timeout_s}"
            )
            return FuturesModelClient(
                self.http_url,
                self.model_name,
                init_timeout_s=init_timeout_s,
                inference_timeout_s=inference_timeout_s,
            )

    _LOGGER.debug(f"Using ports: {ports}")
    with TritonInstance(**ports, model_name="Sleeper", infer_function=_infer_fn) as triton:
        yield triton


# Define a fixture to create and return a Triton server instance
@pytest.fixture(scope="function")
def first_triton_server():
    _LOGGER.debug("Preparing first Triton server.")
    ports = find_free_ports()
    yield from triton_server_builder(ports)


# Define a fixture to create and return a Triton server instance
@pytest.fixture(scope="function")
def second_triton_server():
    _LOGGER.debug("Preparing second Triton server.")
    ports = find_free_ports()
    yield from triton_server_builder(ports)


@pytest.fixture(scope="function")
async def first_async_http_client(first_triton_server):
    _LOGGER.debug(
        f"Preparing client for {first_triton_server.http_url} with init timeout {_GARGANTUAN_TIMEOUT} and inference timeout {_SMALL_TIMEOUT}."
    )
    yield first_triton_server.get_model_futures_client()


@pytest.fixture(scope="function")
async def second_async_http_client(second_triton_server):
    _LOGGER.debug(
        f"Preparing client for {second_triton_server.http_url} with init timeout {_GARGANTUAN_TIMEOUT} and inference timeout {_SMALL_TIMEOUT}."
    )
    yield second_triton_server.get_model_futures_client()


@pytest.fixture(scope="session")
def input_sleep_large():
    _LOGGER.debug(f"Preparing input array with value {_LARGE_TIMEOUT}.")
    yield np.array([[_LARGE_TIMEOUT]], dtype=np.float64)


def test_infer_sample_success_futures(first_async_http_client, second_async_http_client, input_sleep_large):
    _LOGGER.debug(f"Testing async grpc_client with input {input_sleep_large}.")
    with first_async_http_client as first_client:
        with second_async_http_client as second_client:
            first_future = first_client.infer_sample(input_sleep_large)
            second_future = second_client.infer_sample(input_sleep_large)
            done, _not_done = wait([first_future, second_future], timeout=_LARGE_TIMEOUT * 1.3)
            assert len(done) == 2
            first = first_future.result()
            second = second_future.result()
            assert first["OUTPUT_1"] == input_sleep_large
            assert second["OUTPUT_1"] == input_sleep_large
