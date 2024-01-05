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
"""Module handling communication between RequestsServer and RequestsServerClients."""
import asyncio
import enum
import functools
import json
import logging
import pathlib
import socket
import threading
import time
import traceback
import typing
import uuid

import zmq  # pytype: disable=import-error
import zmq.asyncio  # pytype: disable=import-error

LOGGER = logging.getLogger(__name__)
SERVER_LOGGER = LOGGER.getChild("server")
CLIENT_LOGGER = LOGGER.getChild("client")

_STARTUP_TIMEOUT_S = 1.0


class PyTritonResponseFlags(enum.IntFlag):
    """Response flags for PyTritonInferenceHandler."""

    EOS = enum.auto()  # End Of Stream
    ERROR = enum.auto()


class _RequestsServerState(enum.Enum):
    STOPPED = enum.auto()
    STARTING = enum.auto()
    STARTED = enum.auto()
    STOPPING = enum.auto()


def _set_current_task_name(name: str):
    current_task = asyncio.current_task()
    if current_task is not None:
        current_task.set_name(name)


_RequestScope = typing.Dict[str, typing.Any]
_HandleRequestsCoro = typing.Callable[[_RequestScope, bytes, zmq.asyncio.Socket], typing.Awaitable[typing.Any]]
HandleResponsesCoro = typing.Callable[[_RequestScope, asyncio.Queue], typing.Awaitable[typing.Any]]


class RequestsServer:
    """Class for serving available inference requests and passing inference responses."""

    def __init__(self, url: str, handle_responses_fn: HandleResponsesCoro):
        """Initialize RequestsServer.

        Args:
            url: url to bind socket
            handle_responses_fn: couroutine handling responses from InferenceHandler
        """
        self._url = url
        self._handle_responses_fn = handle_responses_fn
        self._state = _RequestsServerState.STOPPED
        self._state_condition = threading.Condition()
        self._shutdown_event = asyncio.Event()  # TODO: is it still required having condition?
        self._server_loop = None

        # requests_id -> results asyncio.Queue map
        self._responses_queues: typing.Dict[bytes, asyncio.Queue] = {}
        self._handle_responses_tasks: typing.Dict[bytes, asyncio.Task] = {}

    def run(self):
        """Run RequestsServer.

        It stops when handle_messages coroutine finishes.

        Raises:
            RuntimeError: if RequestsServer is already running
        """
        with self._state_condition:
            if self._state != _RequestsServerState.STOPPED:
                raise RuntimeError(f"Cannot run {type(self).__name__} as it is already running")

            self._state = _RequestsServerState.STARTING
            self._state_condition.notify_all()

        assert len(self._responses_queues) == 0
        assert len(self._handle_responses_tasks) == 0

        asyncio.run(self.handle_messages())

    @property
    def server_loop(self) -> typing.Optional[asyncio.AbstractEventLoop]:
        """Get asyncio loop for RequestsServer.

        Returns:
            asyncio.AbstractEventLoop: asyncio loop for RequestsServer or None if server is not started yet
        """
        return self._server_loop

    def wait_till_running(self):
        """Wait till RequestsServer is running.

        Raises:
            RuntimeError: if RequestsServer is shutting down or not launched yet
        """
        with self._state_condition:
            if self._state == _RequestsServerState.STARTING:
                self._state_condition.wait_for(
                    lambda: self._state == _RequestsServerState.STARTED, timeout=_STARTUP_TIMEOUT_S
                )
            elif self._state == _RequestsServerState.STOPPED:
                raise RuntimeError("Cannot push requests before RequestsServer is started")
            elif self._state == _RequestsServerState.STOPPING:
                raise RuntimeError(f"Cannot push requests while {type(self).__name__} is shutting down")

    def push(self, requests_id: bytes, requests_payload: bytes):
        """Push requests to InferenceHandler.

        Args:
            requests_id: id of requests
            requests_payload: payload of requests

        Returns:
            asyncio.Task: task handling responses from InferenceHandler

        Raises:
            RuntimeError: if RequestsServer is shutting down or not launched yet
        """
        self.wait_till_running()
        return asyncio.run_coroutine_threadsafe(self.send_requests(requests_id, requests_payload), self.server_loop)

    async def handle_messages(self):
        """Coroutine for handling messages from InferenceHandler."""
        self._server_loop = asyncio.get_running_loop()
        try:
            SERVER_LOGGER.debug(f"Binding socket to url='{self._url}'")
            self._zmq_context = zmq.asyncio.Context()
            self._socket = self._zmq_context.socket(zmq.DEALER)
            self._socket.bind(self._url)
        except (TypeError, zmq.error.ZMQError) as e:
            raise ValueError(
                f"Error occurred during binding socket to url='{self._url}' (e: {e})." "RequestsServer will be closed."
            ) from e

        _set_current_task_name("handle_messages")

        with self._state_condition:
            if self._state != _RequestsServerState.STARTING:
                self._state = _RequestsServerState.STOPPED
                self._state_condition.notify_all()
                raise RuntimeError(f"Cannot start {type(self).__name__} as it is not in STARTING state")

            self._state = _RequestsServerState.STARTED
            self._state_condition.notify_all()

        def _all_responses_processed():
            return not any([self._handle_responses_tasks, self._responses_queues])

        try:
            flag_check_interval_s = 1.0
            # have to receive mssages untill all requestss to be processed, despite shutdown event is set
            while not self._shutdown_event.is_set() or not _all_responses_processed():
                requests_id = b"<unknown>"
                try:
                    requests_id, flags, responses_payload = await asyncio.wait_for(
                        self._socket.recv_multipart(), flag_check_interval_s
                    )
                    SERVER_LOGGER.debug(f"Received response {requests_id.hex()} {flags.hex()} {responses_payload}")
                    flags = int.from_bytes(flags, byteorder="big")
                    responses_queue = self._responses_queues[requests_id]
                    responses_queue.put_nowait((flags, responses_payload))  # queue have no max_size
                except asyncio.TimeoutError:
                    continue
                except KeyError:
                    SERVER_LOGGER.warning(f"Received response for unknown requests {requests_id.hex()}. Ignoring it.")
                except asyncio.CancelledError:
                    SERVER_LOGGER.info("Received CancelledError")
                    self._shutdown_event.set()
        finally:
            # Received all responses, close socket
            SERVER_LOGGER.debug("Closing socket")
            try:
                if self._socket is not None:
                    self._socket.close(linger=0)
                    self._socket = None
            except zmq.error.ZMQError as e:
                SERVER_LOGGER.error(f"Error occurred during closing socket (e: {e}).")

            try:
                if self._zmq_context is not None:
                    self._zmq_context.term()
                    self._zmq_context = None
            except zmq.error.ZMQError as e:
                SERVER_LOGGER.error(f"Error occurred during closing zmq context (e: {e}).")

            self._server_loop = None

            with self._state_condition:
                self._state = _RequestsServerState.STOPPED
                self._state_condition.notify_all()

            SERVER_LOGGER.debug("Socket for handle_messages task closed")
            self._shutdown_event.clear()
            SERVER_LOGGER.debug(f"Leaving handle_messages task from {type(self).__name__}")

    def shutdown(self):
        """Close RequestsServer.

        Don't wait for handle_messages coroutine to finish.
        """
        SERVER_LOGGER.debug("Closing RequestsServer")
        with self._state_condition:
            self._state = _RequestsServerState.STOPPING
            self._state_condition.notify_all()
        self._shutdown_event.set()

    async def send_requests(self, requests_id: bytes, requests_payload: bytes) -> asyncio.Task:
        """Send requests to InferenceHandler.

        Args:
            requests_id: id of requests
            requests_payload: payload of requests

        Returns:
            asyncio.Task: task handling responses from InferenceHandler

        Raises:
            RuntimeError: if RequestsServer is shutting down or requests_id is already pending
        """
        if self._shutdown_event.is_set():
            SERVER_LOGGER.debug(f"Cannot send requests while {type(self).__name__} is {self._state.name}")
            raise RuntimeError(f"Cannot send requests while {type(self).__name__} is {self._state.name}")

        if requests_id in self._responses_queues or requests_id in self._handle_responses_tasks:
            SERVER_LOGGER.debug(f"Cannot send requests with id {requests_id.hex()} as such id is already pending")
            raise RuntimeError(f"Cannot send requests with id {requests_id.hex()} as such id is already pending")

        _set_current_task_name(f"send_requests-{requests_id.hex()}")

        self._responses_queues[requests_id] = asyncio.Queue()
        scope = {"requests_id": requests_id}
        handle_responses_task = self._server_loop.create_task(
            self._handle_responses(scope, self._responses_queues[requests_id]),
            name=f"handle_responses-{requests_id.hex()}",
        )
        self._handle_responses_tasks[requests_id] = handle_responses_task

        # FIXME: check if can not copy buffers; in case copy=False send_multipart returns MessageTracker
        #       https://pyzmq.readthedocs.io/en/latest/api/zmq.html#zmq.Socket.send_multipart
        #       consider send_pyobject|send_serialized (but it is not multipart)

        # sending in same loop, thus thread as handle_messages
        # send_multipart doesn't return anything, as it copies requests_payload
        await self._socket.send_multipart([requests_id, requests_payload])
        SERVER_LOGGER.debug(f"Sent requests {requests_id.hex()}")

        return handle_responses_task

    async def _handle_responses(self, scope, responses_queue: asyncio.Queue):
        """Handle responses from InferenceHandler.

        Args:
            scope: scope for handling responses
            responses_queue: queue with responses payload from InferenceHandler
        """
        requests_id = scope["requests_id"]
        SERVER_LOGGER.debug(f"Started handling responses {requests_id.hex()}")
        try:
            return await self._handle_responses_fn(scope, responses_queue)
        finally:
            self._responses_queues.pop(requests_id)
            self._handle_responses_tasks.pop(requests_id)
            SERVER_LOGGER.debug(f"Finished handling responses {requests_id.hex()}")


class RequestsServerClient:
    """RequestsServer client for handling requests from RequestsServer and sending back responses."""

    def __init__(self, url: str, handle_requests_fn: _HandleRequestsCoro, name: typing.Optional[str] = None):
        """Initialize RequestsServerClient.

        Args:
            url: url to connect socket
            handle_requests_fn: couroutine handling requests from InferenceHandler
            name: name of RequestsServerClient
        """
        self._shutdown_event = asyncio.Event()
        self._url = url
        self._handle_requests_fn = handle_requests_fn
        self._handle_requests_tasks: typing.Dict[bytes, asyncio.Task] = {}
        self._handle_requests_tasks_condition = asyncio.Condition()
        self._name = name or f"requests_server_client-{uuid.uuid4().hex[-4:]}"
        self._loop = None

    def run(self):
        """Run RequestsServerClient.

        It stops when handle_requests coroutine finishes.
        """
        asyncio.run(self.handle_requests())

    def shutdown(self) -> None:
        """Close RequestsServerClient.

        Don't wait for handle_requests coroutine to finish.
        """
        CLIENT_LOGGER.debug(f"Closing {type(self).__name__} {self._name}")
        self._shutdown_event.set()

    async def handle_requests(self):
        """Coroutine for handling requests from RequestsServer."""
        name = self._name
        _set_current_task_name(name)

        zmq_context = None
        socket = None
        self._loop = asyncio.get_running_loop()
        try:
            CLIENT_LOGGER.debug(f"Connecting {name} to server listening on {self._url}")
            zmq_context = zmq.asyncio.Context()
            socket = zmq_context.socket(zmq.DEALER)
            socket.connect(self._url)

            send = functools.partial(self._send, socket)

            flag_check_interval_s = 1.0
            while True:
                try:
                    requests_id, requests_payloads = await asyncio.wait_for(
                        socket.recv_multipart(), flag_check_interval_s
                    )
                    scope = {"requests_id": requests_id}
                    CLIENT_LOGGER.debug(f"{requests_id.hex()} received requests")
                    handle_requests_task = self._loop.create_task(self._handle_requests(scope, requests_payloads, send))
                    self._handle_requests_tasks[requests_id] = handle_requests_task
                    handle_requests_task.set_name(f"handle_requests-{requests_id.hex()}")
                except asyncio.TimeoutError:
                    if self._shutdown_event.is_set():
                        break
                    continue

            CLIENT_LOGGER.debug("Waiting for handle_requests tasks to finish")
            async with self._handle_requests_tasks_condition:
                await self._handle_requests_tasks_condition.wait_for(lambda: len(self._handle_requests_tasks) == 0)
            CLIENT_LOGGER.debug("All handle_requests tasks finished")

        except zmq.error.ZMQError:
            CLIENT_LOGGER.exception(
                "Connection error occurred during reading requests. " f"{type(self).__name__} will be closed."
            )
            self._shutdown_event.set()
        except Exception:
            CLIENT_LOGGER.exception(f"Internal {type(self).__name__}. " f"{type(self).__name__} will be closed.")
            self._shutdown_event.set()
        finally:
            try:
                socket_close_timeout_ms = 0  # immediate close (drop not sent messages)
                if socket is not None:
                    socket.close(linger=socket_close_timeout_ms)
            except zmq.error.ZMQError as e:
                CLIENT_LOGGER.error(f"Error occurred during closing socket (e: {e}).")

            try:
                if zmq_context is not None:
                    zmq_context.term()
            except zmq.error.ZMQError as e:
                CLIENT_LOGGER.error(f"Error occurred during closing zmq context (e: {e}).")

            CLIENT_LOGGER.debug(f"Socket for {name} closed")
            self._shutdown_event.clear()
            self._loop = None
            CLIENT_LOGGER.debug(f"Leaving {name}")

    @property
    def name(self) -> str:
        """Get name of RequestsServerClient.

        Returns:
            name of RequestsServerClient
        """
        return self._name

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Get asyncio loop for RequestsServerClient.

        Returns:
            asyncio.AbstractEventLoop: asyncio loop for RequestsServerClient
        """
        return self._loop

    async def _handle_requests(self, scope, requests_payload, send):
        try:
            await self._handle_requests_fn(scope, requests_payload, send)
        # except PyTritonUnrecoverableError:
        #     error = traceback.format_exc()
        #     responses = InferenceHandlerResponses(error=error)
        #     CLIENT_LOGGER.error(
        #         "Unrecoverable error thrown during calling model callable. "
        #         "Shutting down Triton Inference Server. "
        #         f"{error}"
        #     )
        #     self.stopped = True
        #     self._notify_proxy_backend_observers(InferenceHandlerEvent.UNRECOVERABLE_ERROR, error)
        #     CLIENT_LOGGER.debug(f"Send response to proxy model for {model_name}.")
        #     send(responses.as_bytes())
        except Exception:
            error = traceback.format_exc()
            flags = PyTritonResponseFlags.ERROR | PyTritonResponseFlags.EOS
            await send(scope, flags, error.encode())
            CLIENT_LOGGER.error(f"Error occurred during handling requests {scope['requests_id'].hex()}\n{error}")
        finally:
            async with self._handle_requests_tasks_condition:
                self._handle_requests_tasks.pop(scope["requests_id"], None)
                self._handle_requests_tasks_condition.notify()
            CLIENT_LOGGER.debug(f"Finished handling requests {scope['requests_id'].hex()}")

    async def _send(self, socket, scope, flags, requests_payload):
        """Send requests to RequestsServer.

        Args:
            socket: socket for sending requests
            scope: scope for sending requests
            flags: flags for sending requests
            requests_payload: payload of requests
        """
        flags = flags.to_bytes(1, "big")
        await socket.send_multipart([scope["requests_id"], flags, requests_payload])


class HandshakeServer(threading.Thread):
    """Handshake server for passing config."""

    def __init__(self, socket_path: pathlib.Path, inference_handler_config) -> None:
        """Initialize HandshakeServer.

        Args:
            socket_path: path to socket
            inference_handler_config: config for InferenceHandler
        """
        super().__init__(daemon=True, name="handshake-server")
        self._socket_path = socket_path
        try:
            self._config_payload = json.dumps(inference_handler_config).encode()
        except TypeError:
            raise ValueError(f"InferenceHandler config is not serializable: {inference_handler_config}")

        self._server = None
        self._error_from_thread = None

    def start(self):
        """Start HandshakeServer.

        Raises:
            RuntimeError: if HandshakeServer is already running or error occurred during starting
        """
        if self._server:
            raise RuntimeError("HandshakeServer is already running")

        super().start()
        while self._server is None and not self._error_from_thread:
            time.sleep(0.001)
        if self._error_from_thread is not None:
            raise self._error_from_thread

    def run(self):
        """Run HandshakeServer."""
        asyncio.run(self._run())

    async def _run(self):
        try:
            self._server = await asyncio.start_unix_server(self._handle_request, self._socket_path)
            async with self._server:
                try:
                    await self._server.serve_forever()
                except asyncio.CancelledError:
                    pass
        except Exception as e:
            SERVER_LOGGER.error(f"Error occurred during running handshake server (e: {e})")
            self._error_from_thread = e

    def close(self):
        """Close HandshakeServer."""
        loop = self._server.get_loop()
        loop_tasks = asyncio.all_tasks(loop=loop)
        for task in loop_tasks:
            loop.call_soon_threadsafe(task.cancel)

        self.join()
        SERVER_LOGGER.debug("Closed handshake server")

    async def _handle_request(self, reader, writer):
        peername = writer.get_extra_info("peername")
        try:
            request_name = await asyncio.wait_for(reader.readuntil(b"\n"), timeout=1.0)

            if request_name == b"get_config\n":
                writer.write(len(self._config_payload).to_bytes(4, "big"))
                writer.write(self._config_payload)
                await writer.drain()
            else:
                SERVER_LOGGER.warning(f"Unknown request {request_name} from {peername}")

        except asyncio.TimeoutError:
            SERVER_LOGGER.debug(f"Timeout occurred during handling request from {peername}")
        except Exception as e:
            SERVER_LOGGER.error(f"Error occurred during handling request from {peername} (e: {e})")
        finally:
            writer.close()
            await writer.wait_closed()


def get_config_from_handshake_server(socket_path: pathlib.Path, timeout_s: float = 1.0) -> dict:
    """Get config from handshake server.

    Args:
        socket_path: path to socket

    Returns:
        config from handshake server

    Raises:
        TimeoutError: if timeout occurred while waiting for the response
        ValueError: if invalid JSON response from the server
    """
    should_stop_before_s = time.time() + timeout_s
    sock = None
    try:
        LOGGER.debug(f"Waiting for config file {socket_path}")
        while not socket_path.exists() and time.time() < should_stop_before_s:
            time.sleep(0.001)

        if not socket_path.exists():
            raise TimeoutError(f"Timeout occurred while waiting for config file {socket_path}")

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(max(0.0, should_stop_before_s - time.time()))
        sock.connect(socket_path.as_posix())
        sock.sendall(b"get_config\n")

        sock.settimeout(max(0.0, should_stop_before_s - time.time()))
        payload_size = sock.recv(4)
        payload_size = int.from_bytes(payload_size, "big")

        sock.settimeout(max(0.0, should_stop_before_s - time.time()))
        config_payload = sock.recv(payload_size)
        config = json.loads(config_payload)
        return config
    except socket.timeout as e:
        raise TimeoutError(f"Timeout occurred while waiting for config file {socket_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON response from the server.") from e
    finally:
        if sock is not None:
            sock.close()
