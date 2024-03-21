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

import contextlib
import logging
import socket
import time

import numpy as np
import pytest

from pytriton.client import FuturesModelClient, ModelClient
from pytriton.client.exceptions import PyTritonClientInferenceServerError, PyTritonClientTimeoutError
from pytriton.client.utils import create_client_from_url
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import RemoteTriton, Triton, TritonConfig

_LOGGER = logging.getLogger(__name__)

_SMALLEST_TIMEOUT = 0.0
_SMALL_TIMEOUT = 0.5
_GARGANTUAN_TIMEOUT = 10.0


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


# Define a fixture to start and stop the Triton server with the Sleeper model
@pytest.fixture(scope="function")
def triton_server(find_free_ports):
    @batch
    def _infer_fn(**inputs):  # noqa: N803
        _LOGGER.debug(f"Inputs: {inputs}")
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

    _LOGGER.debug(f"Using ports: grpc={find_free_ports}")
    with TritonInstance(**find_free_ports, model_name="LocalIdentity", infer_function=_infer_fn) as triton:
        yield triton


# Define a fixture to create and return a client object with a very small timeout value
@pytest.fixture(scope="function")
def http_client(triton_server):
    _LOGGER.debug(
        f"Preparing client for {triton_server.http_url} with init timeout {_GARGANTUAN_TIMEOUT} and inference timeout {_SMALL_TIMEOUT}."
    )
    yield ModelClient(
        url=triton_server.http_url,
        model_name=triton_server.model_name,
        init_timeout_s=_GARGANTUAN_TIMEOUT,
        inference_timeout_s=_SMALL_TIMEOUT,
    )


# Define a fixture to create and return an input array with a value of 1.5 seconds
@pytest.fixture(scope="session")
def input_sleep_smallest():
    _LOGGER.debug(f"Preparing input array with value {[_SMALLEST_TIMEOUT]}.")
    yield np.array([[_SMALLEST_TIMEOUT]], dtype=np.float64)


def test_remote_triton_not_connected():
    _LOGGER.debug("Testing RemoteTriton not connected after instantiation.")
    t = RemoteTriton(url="localhost:8000")
    assert not t.is_connected()


def test_remote_triton_connect_with_context():
    _LOGGER.debug("Testing RemoteTriton connect with context manager.")
    with Triton() as t:
        assert t.is_connected()
        assert t.is_alive()
        t.bind("m1", lambda: None, [], [])
        assert t.is_alive()
        with RemoteTriton(url="localhost:8000") as rt:
            assert rt.is_connected()
            rt.bind("m2", lambda: None, [], [])
            assert rt.is_alive()


def test_remote_triton_connect():
    _LOGGER.debug("Testing RemoteTriton connect with connect method.")
    with Triton() as t:
        assert t.is_alive()
        assert t.is_connected()

        rt2 = RemoteTriton(url="localhost:8000")
        assert not rt2.is_connected()
        assert rt2.is_alive()
        rt2.bind("m2", lambda: None, [], [])
        assert not rt2.is_connected()
        assert not rt2.is_alive()
        rt2.connect()
        assert rt2.is_connected()
        assert rt2.is_alive()

        with ModelClient("localhost:8000", "m2", lazy_init=False) as _:
            pass

        rt2.stop()

        with create_client_from_url("localhost:8000") as tr_client:
            assert not tr_client.is_model_ready("m2")


def test_bind_multiple_models():
    _LOGGER.debug("Testing Triton bind multiple models.")
    t = Triton()
    assert not t.is_connected()
    assert not t.is_alive()
    t.bind("m1", lambda: None, [], [])
    assert not t.is_alive()
    t.run()
    assert t.is_alive()
    with ModelClient("localhost:8000", "m1", lazy_init=False) as _:
        pass

    t.bind("m2", lambda: None, [], [])
    assert t.is_alive()
    with ModelClient("localhost:8000", "m2", lazy_init=False) as _:
        pass

    t.stop()
    assert not t.is_alive()
    assert not t.is_connected()


def test_local_and_remote_models_context_manager(triton_server, http_client, input_sleep_smallest):
    _LOGGER.debug(f"Testing http_client with input {input_sleep_smallest}.")
    with http_client as local_client:
        result = local_client.infer_sample(input_sleep_smallest)
    assert result["OUTPUT_1"] == input_sleep_smallest

    @batch
    def _infer_fn(**inputs):  # noqa: N803
        _LOGGER.debug(f"Inputs: {inputs}")
        return_value = {
            "OUTPUT_1": inputs["INPUT_1"],
        }
        _LOGGER.debug(f"Return value {return_value}")
        return return_value

    remote_model = "RemoteIdentity"

    with RemoteTriton(url=triton_server.http_url) as triton:
        triton.bind(
            model_name=remote_model,
            infer_func=_infer_fn,
            inputs=[
                Tensor(dtype=np.float64, shape=(-1, 1)),
            ],
            outputs=[
                Tensor(dtype=np.float64, shape=(-1, 1)),
            ],
            config=ModelConfig(max_batch_size=128),
        )
        with ModelClient(
            url=triton_server.http_url,
            model_name=remote_model,
            init_timeout_s=_GARGANTUAN_TIMEOUT,
            inference_timeout_s=_GARGANTUAN_TIMEOUT,
            lazy_init=False,
        ) as remote_client:
            result = remote_client.infer_sample(input_sleep_smallest)
            assert result["OUTPUT_1"] == input_sleep_smallest


def test_local_and_remote_models_explicite_run(triton_server, http_client, input_sleep_smallest):
    _LOGGER.debug(f"Testing http_client with input {input_sleep_smallest}.")
    with http_client as local_client:
        result = local_client.infer_sample(input_sleep_smallest)
    assert result["OUTPUT_1"] == input_sleep_smallest

    @batch
    def _infer_fn(**inputs):  # noqa: N803
        _LOGGER.debug(f"Inputs: {inputs}")
        return_value = {
            "OUTPUT_1": inputs["INPUT_1"],
        }
        _LOGGER.debug(f"Return value {return_value}")
        return return_value

    remote_model = "RemoteIdentity"

    triton = RemoteTriton(url=triton_server.http_url)
    triton.connect()
    triton.bind(
        model_name=remote_model,
        infer_func=_infer_fn,
        inputs=[
            Tensor(dtype=np.float64, shape=(-1, 1)),
        ],
        outputs=[
            Tensor(dtype=np.float64, shape=(-1, 1)),
        ],
        config=ModelConfig(max_batch_size=128),
    )
    with ModelClient(
        url=triton_server.http_url,
        model_name=remote_model,
        init_timeout_s=_GARGANTUAN_TIMEOUT,
        inference_timeout_s=_GARGANTUAN_TIMEOUT,
        lazy_init=False,
    ) as remote_client:
        result = remote_client.infer_sample(input_sleep_smallest)
        assert result["OUTPUT_1"] == input_sleep_smallest
    triton.stop()


def test_local_and_remote_models_survive_remote_close(triton_server, http_client, input_sleep_smallest):
    _LOGGER.debug(f"Testing http_client with input {input_sleep_smallest}.")
    with http_client as local_client:
        result = local_client.infer_sample(input_sleep_smallest)
        assert result["OUTPUT_1"] == input_sleep_smallest

        @batch
        def _infer_fn(**inputs):  # noqa: N803
            _LOGGER.debug(f"Inputs: {inputs}")
            return_value = {
                "OUTPUT_1": inputs["INPUT_1"],
            }
            _LOGGER.debug(f"Return value {return_value}")
            return return_value

        remote_model = "RemoteIdentity"

        with RemoteTriton(url=triton_server.http_url) as triton:
            triton.bind(
                model_name=remote_model,
                infer_func=_infer_fn,
                inputs=[
                    Tensor(dtype=np.float64, shape=(-1, 1)),
                ],
                outputs=[
                    Tensor(dtype=np.float64, shape=(-1, 1)),
                ],
                config=ModelConfig(max_batch_size=128),
            )
            with ModelClient(
                url=triton_server.http_url,
                model_name=remote_model,
                init_timeout_s=_GARGANTUAN_TIMEOUT,
                inference_timeout_s=_GARGANTUAN_TIMEOUT,
                lazy_init=False,
            ) as remote_client:
                result = remote_client.infer_sample(input_sleep_smallest)
                assert result["OUTPUT_1"] == input_sleep_smallest
        result = local_client.infer_sample(input_sleep_smallest)
        assert result["OUTPUT_1"] == input_sleep_smallest
        with create_client_from_url(triton_server.http_url) as tr_client:
            assert not tr_client.is_model_ready(remote_model)


def test_local_and_remote_models_closes_client(triton_server, http_client, input_sleep_smallest):
    _LOGGER.debug(f"Testing http_client with input {input_sleep_smallest}.")
    remote_client = None
    with http_client as local_client:
        result = local_client.infer_sample(input_sleep_smallest)
        assert result["OUTPUT_1"] == input_sleep_smallest

    @batch
    def _infer_fn(**inputs):  # noqa: N803
        _LOGGER.debug(f"Inputs: {inputs}")
        return_value = {
            "OUTPUT_1": inputs["INPUT_1"],
        }
        _LOGGER.debug(f"Return value {return_value}")
        return return_value

    remote_model = "RemoteIdentity"

    with RemoteTriton(url=triton_server.http_url) as triton:
        triton.bind(
            model_name=remote_model,
            infer_func=_infer_fn,
            inputs=[
                Tensor(dtype=np.float64, shape=(-1, 1)),
            ],
            outputs=[
                Tensor(dtype=np.float64, shape=(-1, 1)),
            ],
            config=ModelConfig(max_batch_size=128),
        )
        remote_client = ModelClient(
            url=triton_server.http_url,
            model_name=remote_model,
            init_timeout_s=_GARGANTUAN_TIMEOUT,
            inference_timeout_s=_GARGANTUAN_TIMEOUT,
            lazy_init=False,
        )
        result = remote_client.infer_sample(input_sleep_smallest)
        assert result["OUTPUT_1"] == input_sleep_smallest

    with create_client_from_url(triton_server.http_url) as tr_client:
        assert not tr_client.is_model_ready(remote_model)

    with pytest.raises(PyTritonClientInferenceServerError):
        remote_client.infer_sample(input_sleep_smallest)

    with pytest.raises(PyTritonClientTimeoutError):
        remote_client_for_dead_model = ModelClient(
            url=triton_server.http_url,
            model_name=remote_model,
            init_timeout_s=_GARGANTUAN_TIMEOUT,
            inference_timeout_s=_GARGANTUAN_TIMEOUT,
            lazy_init=False,
        )

    with ModelClient(
        url=triton_server.http_url,
        model_name=remote_model,
        init_timeout_s=_GARGANTUAN_TIMEOUT,
        inference_timeout_s=_GARGANTUAN_TIMEOUT,
    ) as remote_client_for_dead_model:
        with pytest.raises(PyTritonClientTimeoutError):
            remote_client_for_dead_model.infer_sample(input_sleep_smallest)


@pytest.mark.skip(
    reason="there is no guarantee that the inference will be sent before the model is unloaded. This test is flaky."
)
def test_local_and_remote_models_inflight_requests(triton_server, http_client, input_sleep_smallest):
    _LOGGER.debug(f"Testing http_client with input {input_sleep_smallest}.")
    with http_client as local_client:
        result = local_client.infer_sample(input_sleep_smallest)
        assert result["OUTPUT_1"] == input_sleep_smallest

        @batch
        def _infer_fn(**inputs):  # noqa: N803
            _LOGGER.debug(f"Inputs: {inputs}")
            sleep_time = 5.0
            _LOGGER.info(f"Will sleep {sleep_time}s")
            time.sleep(sleep_time)
            return_value = {
                "OUTPUT_1": inputs["INPUT_1"],
            }
            _LOGGER.debug(f"Return value {return_value}")
            return return_value

        remote_model = "RemoteIdentity"

        futures_client = None
        result_future = None

        with RemoteTriton(url=triton_server.http_url) as triton:
            triton.bind(
                model_name=remote_model,
                infer_func=_infer_fn,
                inputs=[
                    Tensor(dtype=np.float64, shape=(-1, 1)),
                ],
                outputs=[
                    Tensor(dtype=np.float64, shape=(-1, 1)),
                ],
                config=ModelConfig(max_batch_size=128),
            )
            futures_client = FuturesModelClient(
                url=triton_server.http_url,
                model_name=remote_model,
                init_timeout_s=_GARGANTUAN_TIMEOUT,
                inference_timeout_s=_GARGANTUAN_TIMEOUT,
            )
            result_future = futures_client.infer_sample(input_sleep_smallest)

        result = local_client.infer_sample(input_sleep_smallest)
        assert result["OUTPUT_1"] == input_sleep_smallest

    # model waits until all requests are handled
    result = result_future.result()
    assert result["OUTPUT_1"] == input_sleep_smallest

    if futures_client:
        futures_client.close()


def test_local_and_remote_models_name_clash(triton_server, http_client, input_sleep_smallest):
    _LOGGER.debug(f"Testing http_client with input {input_sleep_smallest}.")
    i_was_called = False

    @batch
    def _infer_fn(**inputs):  # noqa: N803
        _LOGGER.debug(f"Inputs: {inputs}")
        return_value = {
            "OUTPUT_1": inputs["INPUT_1"],
        }
        _LOGGER.debug(f"Return value {return_value}")
        nonlocal i_was_called
        i_was_called = True
        return return_value

    remote_model = triton_server.model_name

    with RemoteTriton(url=triton_server.http_url) as triton:
        triton.bind(
            model_name=remote_model,
            infer_func=_infer_fn,
            inputs=[
                Tensor(dtype=np.float64, shape=(-1, 1)),
            ],
            outputs=[
                Tensor(dtype=np.float64, shape=(-1, 1)),
            ],
            config=ModelConfig(max_batch_size=128),
        )
        with ModelClient(
            url=triton_server.http_url,
            model_name=remote_model,
            init_timeout_s=_GARGANTUAN_TIMEOUT,
            inference_timeout_s=_GARGANTUAN_TIMEOUT,
            lazy_init=False,
        ) as remote_client:
            result = remote_client.infer_sample(input_sleep_smallest)
            assert result["OUTPUT_1"] == input_sleep_smallest
    assert i_was_called

    with create_client_from_url(triton_server.http_url) as tr_client:
        assert not tr_client.is_model_ready(remote_model)

    with http_client as local_client:
        with pytest.raises(PyTritonClientTimeoutError):
            local_client.infer_sample(input_sleep_smallest)
