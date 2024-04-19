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
"""

- client.connect doesn't raise any error on missing endpoint
- recv_requests task tries to recv data and waits for stop event

"""

import asyncio
import logging
import pathlib
import queue
import threading
import time
from concurrent.futures import Future as ConcurrentFuture

import pytest

from pytriton.proxy.communication import (
    HandshakeServer,
    RequestsServer,
    RequestsServerClient,
    get_config_from_handshake_server,
)

LOGGER = logging.getLogger(__name__)


def test_client_run_without_server(tmp_path: pathlib.Path):
    """Client should not raise error on missing server endpoint."""

    url = f"ipc://{tmp_path}/test_client_run_without_server"

    async def _handle_requests_fn(scope, requests_payload: bytes, send):
        pass

    requests_client = RequestsServerClient(url, handle_requests_fn=_handle_requests_fn)

    client_thread = threading.Thread(target=requests_client.run, name="requests_client")
    client_thread.start()
    time.sleep(1)
    requests_client.shutdown()
    client_thread.join()


def test_server_run():
    url = "inproc://test_server_run"

    async def _handle_responses_fn(scope, queue, responses_future):
        pass

    requests_server = RequestsServer(url, handle_responses_fn=_handle_responses_fn)

    server_thread = threading.Thread(target=requests_server.run, name="requests_server")
    server_thread.start()
    time.sleep(1)
    requests_server.shutdown()
    server_thread.join()


def test_server_and_client_start_stop(tmp_path: pathlib.Path):
    url = f"ipc://{tmp_path}/test_server_and_client_start_stop"

    async def _handle_responses_fn(scope, queue, responses_future):
        pass

    async def _handle_requests_fn(scope, requests_payload: bytes, send):
        pass

    requests_server = RequestsServer(url, handle_responses_fn=_handle_responses_fn)
    requests_client = RequestsServerClient(url, handle_requests_fn=_handle_requests_fn)

    server_thread = threading.Thread(target=requests_server.run, name="requests_server")
    server_thread.start()
    client_thread = threading.Thread(target=requests_client.run, name="requests_client")
    client_thread.start()
    time.sleep(1)
    requests_client.shutdown()
    requests_server.shutdown()
    client_thread.join()
    server_thread.join()


def test_server_and_client_run(tmp_path: pathlib.Path):
    url = f"ipc://{tmp_path}/test_server_and_client_run"

    final_queue = queue.Queue()

    # new task for each response
    async def _handle_responses_fn(scope, responses_queue, responses_future):
        requests_id = scope["requests_id"]
        flags, responses_payload = await responses_queue.get()
        final_queue.put((requests_id, flags, responses_payload))
        responses_future.set_result(None)

    # new task for each request
    async def _passthrough_request_fn(scope, requests_payload: bytes, send):
        responses_payload = requests_payload
        flags = 0
        await send(scope, flags, responses_payload)

    requests_server = RequestsServer(url, handle_responses_fn=_handle_responses_fn)
    requests_client = RequestsServerClient(url, handle_requests_fn=_passthrough_request_fn)

    client_thread = None
    server_thread = None
    try:
        server_thread = threading.Thread(target=requests_server.run, name="requests_server")
        server_thread.start()
        client_thread = threading.Thread(target=requests_client.run, name="requests_client")
        client_thread.start()

        future = ConcurrentFuture()
        requests_server.wait_till_running()
        handle_responses_task_future0 = asyncio.run_coroutine_threadsafe(
            requests_server.send_requests(b"0x00", b"0", future), requests_server.server_loop
        )
        future = ConcurrentFuture()
        handle_responses_task_future1 = asyncio.run_coroutine_threadsafe(
            requests_server.send_requests(b"0x01", b"1", future), requests_server.server_loop
        )

        handle_responses_task0 = handle_responses_task_future0.result()
        handle_responses_task1 = handle_responses_task_future1.result()

    finally:
        requests_client.shutdown()
        requests_server.shutdown()
        if client_thread:
            client_thread.join()
        if server_thread:
            server_thread.join()

    # when server is shutdown, the handle responses tasks results should be available
    assert handle_responses_task0.result() is None
    assert handle_responses_task1.result() is None

    assert final_queue.get() == (b"0x00", 0, b"0")
    assert final_queue.get() == (b"0x01", 0, b"1")
    assert final_queue.empty()


def test_handshake_server_client(tmp_path: pathlib.Path):
    socket_path = tmp_path / "handshake.socket"

    server = None
    try:
        server = HandshakeServer(socket_path, {})
        server.start()
    finally:
        if server is not None:
            server.close()
            assert not server.is_alive()


def test_handshake_server_raises_error_on_config_non_serializable(tmp_path: pathlib.Path):
    socket_path = tmp_path / "handshake.socket"

    server = None
    try:
        with pytest.raises(ValueError):
            server = HandshakeServer(socket_path, {"foo": lambda x: x})
    finally:
        if server is not None:
            server.close()
            assert not server.is_alive()


@pytest.mark.xfail(
    reason="TODO: OSError: [Errno 98] Address already in use is not raised during asyncio.start_unix_server on same socket path"
)
def test_handshake_server_raises_error_on_try_to_bind_twice(tmp_path: pathlib.Path):
    socket_path = tmp_path / "handshake.socket"

    server1 = None
    server2 = None
    try:
        server1 = HandshakeServer(socket_path, {})
        server1.start()

        server2 = HandshakeServer(socket_path, {})
        with pytest.raises(RuntimeError):
            server2.start()
    finally:
        if server1 is not None:
            server1.close()
            assert not server1.is_alive()

        if server2 is not None:
            server2.close()
            assert not server2.is_alive()


def test_handshake_server_start_raises_error_on_double_start(tmp_path: pathlib.Path):
    socket_path = tmp_path / "handshake.socket"

    server = None
    try:
        server = HandshakeServer(socket_path, {})
        server.start()
        with pytest.raises(RuntimeError):
            server.start()
    finally:
        if server is not None:
            server.close()
            assert not server.is_alive()


def test_handshake_server_client_with_client(tmp_path: pathlib.Path):
    socket_path = tmp_path / "handshake.socket"
    config = {"foo": "bar"}
    server = None
    client = None
    try:
        server = HandshakeServer(socket_path, config)
        server.start()
        obtained_config = get_config_from_handshake_server(socket_path)
        assert obtained_config == config
    finally:
        if server is not None:
            server.close()
            assert not server.is_alive()
        if client is not None:
            client.close()


def test_handshake_client_timeout_on_no_server(tmp_path: pathlib.Path):
    socket_path = tmp_path / "handshake.socket"

    with pytest.raises(TimeoutError):
        get_config_from_handshake_server(socket_path)


def test_handshake_client_timeout_on_slow_server(tmp_path: pathlib.Path, mocker):
    socket_path = tmp_path / "handshake.socket"

    async def slow_handle_request(reader, writer):
        await asyncio.sleep(0.5)

    server = None
    try:
        mocker.patch.object(HandshakeServer, "_handle_request", side_effect=slow_handle_request)
        server = HandshakeServer(socket_path, {})
        server.start()

        with pytest.raises(TimeoutError):
            get_config_from_handshake_server(socket_path, timeout_s=0.1)
    finally:
        if server is not None:
            server.close()
            assert not server.is_alive()
