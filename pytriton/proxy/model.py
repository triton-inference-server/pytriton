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
"""Model definition for Python Backend for PyTriton.

This file is automatically copied during deployment on Triton and should not be modified.
"""
import asyncio
import base64
import json
import logging
import multiprocessing
import pathlib
import threading
import time
import traceback
import typing

import triton_python_backend_utils as pb_utils  # pytype: disable=import-error

from . import communication, data
from .communication import (  # pytype: disable=import-error
    HandleResponsesCoro,
    HandshakeServer,
    PyTritonResponseFlags,
    RequestsServer,
    get_config_from_handshake_server,
)
from .data import PROTOCOL_VERSION, TensorStoreSerializerDeserializer  # pytype: disable=import-error
from .types import Request, Response, ResponsesOrError  # pytype: disable=import-error

LOGGER = logging.getLogger(__name__)


def _update_loggers():
    def get_triton_backend_logger():
        try:
            # https://github.com/triton-inference-server/python_backend/blob/main/src/pb_stub.cc#L1501
            import triton_python_backend_utils as pb_utils  # pytype: disable=import-error

            logger = pb_utils.Logger  # pytype: disable=module-attr
            logger.error = logger.log_error
            logger.warning = logger.log_warn
            logger.info = logger.log_info
            logger.debug = logger.log_verbose
            # do not set log_to_stderr in Backend
        except (ImportError, AttributeError):
            logger = logging.getLogger("backend")
            root_logger = logging.getLogger()
            if root_logger.level <= logging.INFO:
                multiprocessing.util.log_to_stderr(logging.INFO)
        return logger

    logger = get_triton_backend_logger()
    global LOGGER
    LOGGER = logger
    data.LOGGER = logger
    communication.LOGGER = logger
    communication.SERVER_LOGGER = logger


class TritonRequestsServer:
    """Class for handling communication between Triton and Inference Callable."""

    def __init__(
        self,
        url: str,
        responses_handle_fn: HandleResponsesCoro,
        serializer_deserializer,
        model_config: typing.Dict[str, typing.Any],
    ):
        """Create TritonRequestsServer object.

        Args:
            url: url to the socket
            responses_handle_fn: coroutine that handles responses from InferenceHandler
            serializer_deserializer: object that serializes and deserializes requests and responses
            model_config: Triton model config
        """
        self._model_config = model_config
        self._model_inputs_names = [model_input["name"] for model_input in model_config["input"]]
        self._server = RequestsServer(url, responses_handle_fn)
        self._serializer_deserializer = serializer_deserializer

    def run(self):
        """Run requests server.

        This method should be called in separate thread.
        """
        self._server.run()

    def shutdown(self):
        """Shutdown requests server.

        Doesn't wait for server to stop. Should wait till thread running TritonRequestsServer is finished.
        """
        self._server.shutdown()

    def push(self, requests_id: bytes, triton_requests):
        """Push requests to TritonRequestsServer queue.

        Args:
            requests_id: id of requests
            triton_requests: list of Triton requests
        """
        self._server.wait_till_running()  # wait until loop is up and running, raise RuntimeError if server is stopping or not launched yet
        return asyncio.run_coroutine_threadsafe(self._infer(requests_id, triton_requests), self._server.server_loop)

    def _wrap_request(self, triton_request, inputs) -> Request:
        request = {}
        for input_name in inputs:
            input_tensor = pb_utils.get_input_tensor_by_name(triton_request, input_name)
            if input_tensor is not None:
                request[input_name] = input_tensor.as_numpy()
        return Request(data=request, parameters=json.loads(triton_request.parameters()))

    async def _infer(self, requests_id: bytes, triton_requests) -> asyncio.Task:
        wrap_tasks = [
            self._server._server_loop.run_in_executor(
                None, self._wrap_request, triton_request, self._model_inputs_names
            )
            for triton_request in triton_requests
        ]
        requests = await asyncio.gather(*wrap_tasks)
        requests_payload = await self._server.server_loop.run_in_executor(
            None, self._serializer_deserializer.serialize_requests, requests
        )
        # will return when socket.send_multipart returns
        handle_responses_task = await self._server.send_requests(requests_id, requests_payload)
        return handle_responses_task


def _wrap_response(response: Response, requested_outputs_names, model_outputs_dict):
    if response.data is not None:
        only_requested = {key: value for key, value in response.data.items() if key in requested_outputs_names}
        casted = {
            key: value.astype(pb_utils.triton_string_to_numpy(model_outputs_dict[key]["data_type"]))
            for key, value in only_requested.items()
        }
        return pb_utils.InferenceResponse(  # pytype: disable=module-attr
            output_tensors=[
                pb_utils.Tensor(name, value) for name, value in casted.items()  # pytype: disable=module-attr
            ]
        )
    else:
        return None


class BatchResponsesHandler:
    """Class for handling responses from InferenceHandler."""

    def __init__(self, requests_map, serializer_deserializer, model_outputs_dict):
        """Init BatchResponsesHandler object."""
        self._requests_map = requests_map
        self._serializer_deserializer = serializer_deserializer
        self._model_outputs_dict = model_outputs_dict

    async def handle_responses(self, scope: typing.Dict[str, typing.Any], responses_queue: asyncio.Queue):
        """Handle responses from InferenceHandler.

        Args:
            requests_id: id of requests
            responses_queue: queue with responses payload from InferenceHandler

        Returns:
            Triton Responses or TritonModelException
        """
        requests_id: bytes = scope["requests_id"]
        loop = asyncio.get_running_loop()
        triton_requests = self._requests_map[requests_id]

        eos = False
        triton_responses_or_error = None
        while not eos:
            try:
                flags, responses_payload = await responses_queue.get()
                eos = flags & PyTritonResponseFlags.EOS
                error = flags & PyTritonResponseFlags.ERROR

                if error:
                    assert eos
                    triton_responses_or_error = pb_utils.TritonModelException(  # pytype: disable=module-attr
                        responses_payload.decode("utf-8")
                    )
                elif responses_payload:
                    # inference handler should send all responses in payload
                    assert triton_responses_or_error is None
                    responses = await loop.run_in_executor(
                        None, self._serializer_deserializer.deserialize_responses, responses_payload
                    )
                    wrap_tasks = [
                        loop.run_in_executor(
                            None, _wrap_response, response, request.requested_output_names(), self._model_outputs_dict
                        )
                        for request, response in zip(triton_requests, responses)
                    ]
                    triton_responses_or_error = await asyncio.gather(*wrap_tasks)
            except asyncio.CancelledError:
                LOGGER.warning(f"Cancelled responses handler for requests={requests_id.hex()}")
                triton_responses_or_error = pb_utils.TritonModelException(  # pytype: disable=module-attr
                    "Cancelled responses handler"
                )
                eos = True
            finally:
                if not error:
                    await loop.run_in_executor(
                        None, self._serializer_deserializer.free_responses_resources, responses_payload
                    )
                responses_queue.task_done()

        self._requests_map.pop(requests_id)
        return triton_responses_or_error


class DecoupledResponsesHandler:
    """Class for handling responses for decoupled model."""

    def __init__(self, requests_map, serializer_deserializer, model_outputs_dict):
        """Create DecoupledResponsesHandler object."""
        self._requests_map = requests_map
        self._serializer_deserializer = serializer_deserializer
        self._model_outputs_dict = model_outputs_dict

    async def handle_responses(
        self, scope: typing.Dict[str, typing.Any], responses_queue: asyncio.Queue
    ) -> typing.Optional[ResponsesOrError]:
        """Handle responses from InferenceHandler.

        Args:
            requests_id: id of requests
            responses_queue: queue with responses from InferenceHandler

        Returns:
            Responses or None if responses were sent to client
        """
        requests_id: bytes = scope["requests_id"]
        loop = asyncio.get_running_loop()
        triton_requests = self._requests_map[requests_id]
        triton_senders = [request.get_response_sender() for request in triton_requests]

        eos = False
        while not eos:
            try:
                flags, responses_payload = await responses_queue.get()
                LOGGER.debug(f"Got {flags} {responses_payload}")

                eos = flags & PyTritonResponseFlags.EOS
                error = flags & PyTritonResponseFlags.ERROR

                triton_responses = None
                if error:
                    triton_responses = [
                        pb_utils.InferenceResponse(  # pytype: disable=module-attr
                            error=pb_utils.TritonError(responses_payload.decode("utf-8"))  # pytype: disable=module-attr
                        )
                        for _ in triton_senders
                    ]
                else:
                    responses = await loop.run_in_executor(
                        None, self._serializer_deserializer.deserialize_responses, responses_payload
                    )
                    triton_responses = [
                        _wrap_response(response, request.requested_output_names(), self._model_outputs_dict)
                        for request, response in zip(triton_requests, responses)
                    ]

                triton_flags = 0
                if eos:
                    triton_flags = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    triton_responses = triton_responses or [None] * len(triton_senders)

                # run sender.send in parallel in executor
                assert len(triton_responses) == len(triton_senders)
                send_responses_futures = [
                    loop.run_in_executor(None, sender.send, response, triton_flags)
                    for sender, response in zip(triton_senders, triton_responses)
                ]
                await asyncio.gather(*send_responses_futures)
            except asyncio.CancelledError:
                LOGGER.warning(f"Cancelled responses handler for requests={requests_id.hex()}")
                triton_flags = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                triton_response = pb_utils.InferenceResponse(  # pytype: disable=module-attr
                    error=pb_utils.TritonError(error="Cancelled responses handler")  # pytype: disable=module-attr
                )
                send_responses_futures = [
                    loop.run_in_executor(None, sender.send, triton_response, triton_flags) for sender in triton_senders
                ]
                await asyncio.gather(*send_responses_futures)
            finally:
                if not error:
                    await loop.run_in_executor(
                        None, self._serializer_deserializer.free_responses_resources, responses_payload
                    )
                responses_queue.task_done()

        self._requests_map.pop(requests_id)


class TritonInferenceHandlerConfigGenerator:
    """PyTriton Inference handler config generator for Triton PythonBackend."""

    def __init__(self, data_socket: typing.Union[str, pathlib.Path]):
        """Initialize the config generator.

        Args:
            data_socket: path to the data socket
        """
        self._data_socket = pathlib.Path(data_socket)

    def get_config(self) -> typing.Dict[str, typing.Any]:
        """Return the config for the inference handler."""
        return {
            "protocol_version": PROTOCOL_VERSION,
            "data_socket": self._data_socket.as_posix(),
            "authkey": base64.encodebytes(multiprocessing.current_process().authkey).decode("ascii"),
        }


class TritonPythonModel:
    """Triton PythonBackend model implementation for proxy."""

    def __init__(self):
        """Dummy inititializer."""
        self._model_config = None
        self._model_inputs = None
        self._model_outputs = None
        self._model_inputs_names = None
        self._model_instance_name = None
        self._decoupled_model = None
        self._serializer_deserializer = None
        self._requests_server = None
        self._requests_server_thread = None
        self._handshake_server = None
        self._loop = None
        self._frontend = None
        self._requests = None
        self._id_counter = 0

    def initialize(self, args):
        """Triton Inference Server Python Backend API called only once when the model is being loaded.

        Allows the model to initialize any state associated with this model.

        Args:
            args: Dictionary with both keys and values are strings. The dictionary keys and values are:
                * model_config: A JSON string containing the model configuration
                * model_instance_kind: A string containing model instance kind
                * model_instance_device_id: A string containing model instance device ID
                * model_instance_name: A string containing model instance name in form of <model_name>_<instance_group_id>_<instance_id>
                * model_repository: Model repository path
                * model_version: Model version
                * model_name: Model name
        """
        _update_loggers()  # Triton backend logger is available from this point on

        try:
            model_name = args["model_name"]

            self._model_config = model_config_dict = json.loads(args["model_config"])
            self._model_inputs = {model_input["name"] for model_input in model_config_dict["input"]}
            self._model_outputs = {model_output["name"]: model_output for model_output in model_config_dict["output"]}
            self._model_inputs_names = list(self._model_inputs)
            self._decoupled_model = model_config_dict.get("model_transaction_policy", {}).get("decoupled", False)
            self._model_instance_name = args.get("model_instance_name")

            workspace_path = pathlib.Path(model_config_dict["parameters"]["workspace-path"]["string_value"])

            LOGGER.debug(f"Model instance name: {self._model_instance_name}")
            LOGGER.debug(f"Decoupled model: {self._decoupled_model}")
            LOGGER.debug(f"Workspace path: {workspace_path}")
            LOGGER.debug(f"Model inputs: {self._model_inputs}")
            LOGGER.debug(f"Model outputs: {self._model_outputs}")

            # init serializer/deserializer
            data_socket = workspace_path / f"{model_name}-data.sock"
            self._serializer_deserializer = TensorStoreSerializerDeserializer()

            handshake_socket = workspace_path / f"{model_name}-config.sock"
            model_first_instance_name = "_".join(self._model_instance_name.split("_")[:-1] + ["0"])
            if self._model_instance_name == model_first_instance_name:
                inference_handler_config = TritonInferenceHandlerConfigGenerator(data_socket).get_config()
                self._serializer_deserializer.start(data_socket)

                self._handshake_server = HandshakeServer(handshake_socket, inference_handler_config)
                self._handshake_server.start()

            else:
                inference_handler_config = get_config_from_handshake_server(handshake_socket)
                LOGGER.debug(f"Loaded configuration from {handshake_socket}")
                authkey = base64.decodebytes(inference_handler_config["authkey"].encode("ascii"))
                self._serializer_deserializer.connect(data_socket, authkey=authkey)

            self._id_counter = 0
            self._requests = {}

            server_socket_path = workspace_path / f"{self._model_instance_name}-server.sock"
            handler_class = DecoupledResponsesHandler if self._decoupled_model else BatchResponsesHandler
            LOGGER.debug(f"Using {handler_class.__name__} for handling responses")
            self._requests_server = TritonRequestsServer(
                url=f"ipc://{server_socket_path.as_posix()}",
                responses_handle_fn=handler_class(
                    self._requests, self._serializer_deserializer, self._model_outputs
                ).handle_responses,
                serializer_deserializer=self._serializer_deserializer,
                model_config=self._model_config,
            )

            def _run_server():
                _update_loggers()
                self._requests_server.run()

            self._requests_server_thread = threading.Thread(target=_run_server, name="requests-server", daemon=True)
            self._requests_server_thread.start()
        except Exception:
            msg = traceback.format_exc()
            raise pb_utils.TritonModelException(f"Model initialize error: {msg}")  # pytype: disable=module-attr

    def execute(self, triton_requests):
        """Triton Inference Server Python Backend API method.

        Args:
            triton_requests: A list of pb_utils.InferenceRequest

        Returns:
            A list of pb_utils.InferenceResponse. The length of this list is the same as `triton_requests`

        Raises:
            pb_utils.TritonModelException: when model execution fails
        """
        try:

            def _generate_id():
                self._id_counter = (self._id_counter + 1) % 2**32
                return self._id_counter.to_bytes(4, "big")

            requests_id = _generate_id()
            while requests_id in self._requests:
                requests_id = _generate_id()
            self._requests[requests_id] = triton_requests

            # TODO: add this future to container to avoid garbage collection
            handle_responses_task_future = self._requests_server.push(requests_id, triton_requests)

            if not self._decoupled_model:
                handle_responses_task = handle_responses_task_future.result()
                check_interval_s = 0.001
                while not handle_responses_task.done():
                    time.sleep(check_interval_s)
                triton_responses_or_error = handle_responses_task.result()

                if triton_responses_or_error is not None and isinstance(triton_responses_or_error, Exception):
                    raise triton_responses_or_error
            else:
                triton_responses_or_error = None

            return triton_responses_or_error
        except Exception:
            msg = traceback.format_exc()
            raise pb_utils.TritonModelException(f"Model execute error: {msg}")  # pytype: disable=module-attr

    def finalize(self) -> None:
        """Finalize the model cleaning the buffers."""
        LOGGER.debug(f"[{self._model_instance_name}] Finalizing backend instance")
        LOGGER.debug(f"[{self._model_instance_name}] Closing requests server")
        self._requests_server.shutdown()
        self._requests_server_thread.join()

        LOGGER.debug(f"[{self._model_instance_name}] Closing requests/responses serializer/deserializer")
        self._serializer_deserializer.close()
        self._serializer_deserializer = None

        LOGGER.debug(f"[{self._model_instance_name}] Closing handshake server")
        if self._handshake_server:
            self._handshake_server.close()
            self._handshake_server = None

        LOGGER.debug(f"[{self._model_instance_name}] Finalized.")
        self._model_instance_name = None

    @property
    def model_supports_batching(self) -> bool:
        """Return if model supports batching.

        Returns:
            True if model support batching, False otherwise.
        """
        return self._model_config["max_batch_size"] > 0
