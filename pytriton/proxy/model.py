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
"""Model definition for Python Backend.

This file is automatically copied during deployment on Triton.
"""
import base64
import itertools
import json
import traceback
import typing

import triton_python_backend_utils as pb_utils  # pytype: disable=import-error
import zmq  # pytype: disable=import-error

from .communication import InferenceHandlerRequests, InferenceHandlerResponses, MetaRequestResponse, TensorStore
from .types import Request

_ACK = b""


class _ResponsesIterator:
    def __init__(self, socket, requests_number: int) -> None:
        self._socket = socket
        self._requests_eos = [False] * requests_number

    def __iter__(self):
        return self

    def __next__(self):
        responses_payload = self._socket.recv()

        meta_responses = InferenceHandlerResponses.from_bytes(responses_payload)
        if meta_responses.error:
            raise pb_utils.TritonModelException(meta_responses.error)  # pytype: disable=module-attr

        for meta_response in meta_responses.responses:
            self._requests_eos[meta_response.idx] |= meta_response.eos

        if all(self._requests_eos):
            raise StopIteration()
        else:
            self._socket.send(_ACK)  # send ack to receive further results

        return meta_responses


class _ResponsesSender:
    def __init__(self, requests: typing.List):
        self._requests = requests
        self._responses = []  # typing.List[pb_utils.InferenceResponse]

    def send(self, responses: typing.List, eos: typing.Optional[typing.Sequence[bool]] = None):
        assert len(responses) == len(self._requests)
        # here all responses should be not None
        self._responses.extend(responses)

    def finish(self):
        to_send = self._responses
        self._responses = []
        return to_send


class _DecoupledResponsesSender:
    def __init__(self, requests: typing.List):
        self._senders = [request.get_response_sender() for request in requests]
        self._eos = [False] * len(requests)

    def send(self, responses: typing.List, eos: typing.Optional[typing.Sequence[typing.Optional[bool]]] = None):
        assert len(responses) == len(self._senders)
        eos = eos or [None] * len(self._senders)
        for response_idx, (response, response_eos) in enumerate(zip(responses, eos)):

            if response is None and not response_eos:
                continue

            sender = self._senders[response_idx]
            self._eos[response_idx] |= response_eos
            flags = 0
            if response_eos:
                flags |= pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            if response is not None:
                sender.send(response, flags=flags)
            elif response_eos:
                sender.send(flags=flags)

    def finish(self):
        for idx, sender in enumerate(self._senders):
            if not self._eos[idx]:
                sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        return None


class TritonPythonModel:
    """Triton PythonBackend model implementation for proxy."""

    def __init__(self):
        """Create TritonPythonModel object."""
        self.model_config = None
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)

        self.model_config = None
        self.model_inputs = []
        self.model_outputs = []
        self.model_outputs_dict = {}

        self._tensor_store = None
        self._last_response_ids = []

        self._sender_cls = None

    def initialize(self, args):
        """Triton Inference Server Python Backend API called only once when the model is being loaded.

        Allows the model to initialize any state associated with this model.

        Args:
            args: Dictionary with both keys and values are strings. The dictionary keys and values are:
                * model_config: A JSON string containing the model configuration
                * model_instance_kind: A string containing model instance kind
                * model_instance_device_id: A string containing model instance device ID
                * model_repository: Model repository path
                * model_version: Model version
                * model_name: Model name
        """
        logger = pb_utils.Logger  # pytype: disable=module-attr
        try:
            logger.log_verbose("Reading model config")
            self.model_config = json.loads(args["model_config"])
            shared_memory_socket = self.model_config["parameters"]["shared-memory-socket"]["string_value"]
            logger.log_verbose(f"Connecting to IPC socket at {shared_memory_socket}")

            instance_data = self._get_instance_data(shared_memory_socket)
            self.socket.connect(instance_data["shared-memory-socket"])
            logger.log_verbose(f"Connected to socket {shared_memory_socket}.")

            self.model_inputs = self.model_config["input"]
            self.model_outputs = self.model_config["output"]
            self.model_outputs_dict = {output_def["name"]: output_def for output_def in self.model_outputs}

            logger.log_verbose(f"Model inputs: {self.model_inputs}")
            logger.log_verbose(f"Model outputs: {self.model_outputs}")

            data_store_socket = instance_data["data-store-socket"]
            auth_key = base64.b64decode(instance_data["auth-key"])

            self._tensor_store = TensorStore(data_store_socket, auth_key)
            self._tensor_store.connect()
            self._last_response_ids = []

            self._sender_cls = {
                False: _ResponsesSender,
                True: _DecoupledResponsesSender,
            }[self.model_config.get("model_transaction_policy", {}).get("decoupled", False)]

        except Exception:
            msg = traceback.format_exc()
            raise pb_utils.TritonModelException(f"Model initialize error: {msg}")  # pytype: disable=module-attr

    def execute(self, triton_requests):
        """Triton Inference Server Python Backend API method.

        Args:
            triton_requests: A list of pb_utils.InferenceRequest

        Returns:
            A list of pb_utils.InferenceResponse. The length of this list is the same as `triton_requests`
        """
        logger = pb_utils.Logger  # pytype: disable=module-attr

        try:
            meta_requests = self._put_requests_to_buffer(triton_requests)
            logger.log_verbose(f"Sending requests {meta_requests}.")
            self.socket.send(meta_requests.as_bytes())

            responses_iterator = _ResponsesIterator(self.socket, len(triton_requests))
            responses_sender = self._sender_cls(triton_requests)
            logger.log_verbose(f"using sender {responses_sender}")
            for meta_responses in responses_iterator:
                logger.log_verbose(f"Received response: {meta_responses}")
                triton_responses = self._handle_responses_from_buffer(meta_responses, len(triton_requests))
                responses_sender.send(
                    triton_responses, eos=[meta_response.eos for meta_response in meta_responses.responses]
                )

                # TODO: fix leak on error
                self._last_response_ids.extend(
                    itertools.chain(
                        *[
                            meta_response.data.values()
                            for meta_response in meta_responses.responses
                            if meta_response.data is not None
                        ]
                    )
                )
            return responses_sender.finish()
        except pb_utils.TritonModelException:  # pytype: disable=module-attr
            raise
        except Exception:
            msg = traceback.format_exc()
            raise pb_utils.TritonModelException(f"Model execute error: {msg}")  # pytype: disable=module-attr

    def finalize(self) -> None:
        """Finalize the model cleaning the buffers."""
        logger = pb_utils.Logger  # pytype: disable=module-attr
        logger.log_verbose("Finalizing backend instance.")
        logger.log_verbose("Cleaning socket and context.")
        socket_close_timeout_s = 0
        if self.socket:
            self.socket.close(linger=socket_close_timeout_s)
        if self.context:
            self.context.term()
        self.socket = None
        self.context = None

        logger.log_verbose("Removing allocated shared memory.")
        for tensor_id in self._last_response_ids:
            self._tensor_store.release_block(tensor_id)
        self._last_response_ids = []
        self._tensor_store.close()
        self._tensor_store = None

        logger.log_verbose("Finalized.")

    @property
    def model_supports_batching(self) -> bool:
        """Return if model supports batching.

        Returns:
            True if model support batching, False otherwise.
        """
        return self.model_config["max_batch_size"] > 0

    def _get_instance_data(self, shared_memory_socket) -> typing.Dict[str, str]:
        handshake_socket = self.context.socket(zmq.REQ)
        handshake_socket.connect(shared_memory_socket)
        handshake_socket.send_string("get_instance_socket")
        instance_data_payload = handshake_socket.recv()
        handshake_socket.close()
        instance_data = json.loads(instance_data_payload.decode("utf-8"))
        logger = pb_utils.Logger  # pytype: disable=module-attr
        instance_data_copy = instance_data.copy()
        if "auth-key" in instance_data_copy:
            instance_data_copy["auth-key"] = "***"
        logger.log_verbose(f"Obtained instance data: {instance_data_copy}")
        return instance_data

    def _put_requests_to_buffer(self, triton_requests) -> InferenceHandlerRequests:
        logger = pb_utils.Logger  # pytype: disable=module-attr

        while self._last_response_ids:
            tensor_id = self._last_response_ids.pop()
            self._tensor_store.release_block(tensor_id)

        logger.log_verbose("Collecting input data from request.")

        requests = []
        for triton_request in triton_requests:
            request = {}
            for model_input in self.model_inputs:
                input_tensor = pb_utils.get_input_tensor_by_name(triton_request, model_input["name"])
                if input_tensor is not None:
                    request[model_input["name"]] = input_tensor.as_numpy()
            requests.append(Request(data=request, parameters=json.loads(triton_request.parameters())))

        input_arrays_with_coords = [
            (request_idx, input_name, tensor)
            for request_idx, request in enumerate(requests)
            for input_name, tensor in request.items()
        ]
        tensor_ids = self._tensor_store.put([tensor for *_, tensor in input_arrays_with_coords])
        requests_with_ids = [{} for _ in range(len(requests))]
        for (request_idx, input_name, _), tensor_id in zip(input_arrays_with_coords, tensor_ids):
            requests_with_ids[request_idx][input_name] = tensor_id

        return InferenceHandlerRequests(
            requests=[
                MetaRequestResponse(idx=idx, data=request_with_ids, parameters=request.parameters)
                for idx, (request, request_with_ids) in enumerate(zip(requests, requests_with_ids))
            ]
        )

    def _handle_responses_from_buffer(
        self, meta_response: InferenceHandlerResponses, requests_number: int
    ) -> typing.List:
        def _get_array_and_wrap(output_name, tensor_id: str) -> pb_utils.Tensor:  # pytype: disable=module-attr
            output_array = self._tensor_store.get(tensor_id)
            if output_name in self.model_outputs_dict:
                dtype = pb_utils.triton_string_to_numpy(self.model_outputs_dict[output_name]["data_type"])
                output_array = output_array.astype(dtype)
            return pb_utils.Tensor(output_name, output_array)  # pytype: disable=module-attr

        responses = meta_response.responses
        if len(responses) != requests_number:
            raise pb_utils.TritonModelException(  # pytype: disable=module-attr
                f"Number of responses {len(responses)} does not match number of requests {requests_number}"
            )
        triton_inference_responses = [None] * requests_number
        for response in responses:
            if response.data is None:
                continue
            triton_inference_responses[response.idx] = pb_utils.InferenceResponse(  # pytype: disable=module-attr
                [_get_array_and_wrap(output_name, tensor_id) for output_name, tensor_id in response.data.items()]
            )
        return triton_inference_responses
