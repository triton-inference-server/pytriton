# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import json
import traceback

import triton_python_backend_utils as pb_utils  # pytype: disable=import-error
import zmq  # pytype: disable=import-error

from .communication import InferenceHandlerRequest, InferenceHandlerResponse, MetaRequestResponse, ShmManager
from .types import Request, Response


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

        self.shm_request_manager = ShmManager()
        self.shm_response_manager = ShmManager()

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
            instance_shared_memory_socket = self._get_instance_socket(shared_memory_socket)
            self.socket.connect(instance_shared_memory_socket)
            logger.log_verbose(f"Connected to socket {shared_memory_socket}.")

            self.model_inputs = self.model_config["input"]
            self.model_outputs = self.model_config["output"]
            self.model_outputs_dict = {output_def["name"]: output_def for output_def in self.model_outputs}

            logger.log_verbose(f"Model inputs: {self.model_inputs}")
            logger.log_verbose(f"Model outputs: {self.model_outputs}")

        except Exception:
            msg = traceback.format_exc()
            raise pb_utils.TritonModelException(f"Model initialize error: {msg}")  # pytype: disable=module-attr

    def execute(self, requests):
        """Triton Inference Server Python Backend API method.

        Method concatenate requests data into single batch and sends to backend client.
        Depending on the batching configuration (e.g. Dynamic Batching) used,
        `requests` may contain multiple requests.

        Args:
            requests: A list of pb_utils.InferenceRequest

        Returns:
            A list of pb_utils.InferenceResponse. The length of this list is the same as `requests`
        """
        if not self.model_supports_batching and len(requests) > 1:
            raise RuntimeError(
                "Code assumes that Triton doesn't put multiple requests when model doesn't support batching"
            )
        logger = pb_utils.Logger  # pytype: disable=module-attr
        logger.log_verbose("Collecting input data from request.")
        # input_idx, merged_batch_size, ...
        inputs = []

        for request in requests:
            request_dict = {}
            for model_input in self.model_inputs:
                input_tensor = pb_utils.get_input_tensor_by_name(request, model_input["name"])
                if input_tensor is not None:
                    request_dict[model_input["name"]] = input_tensor.as_numpy()
            inputs.append(Request(data=request_dict, parameters=json.loads(request.parameters())))

        # input_idx, merged_batch_size, ...
        output_array = self._exec_requests(inputs, logger)

        responses = []
        for resp in output_array:
            output_tensors = []
            for output_name, output in resp.items():
                if output_name in self.model_outputs_dict:
                    dtype = pb_utils.triton_string_to_numpy(self.model_outputs_dict[output_name]["data_type"])
                    output = output.astype(dtype)
                tensor = pb_utils.Tensor(output_name, output)  # pytype: disable=module-attr
                output_tensors.append(tensor)

            response = pb_utils.InferenceResponse(output_tensors=output_tensors)  # pytype: disable=module-attr
            responses.append(response)

        return responses

    def finalize(self) -> None:
        """Finalize the model cleaning the buffers."""
        logger = pb_utils.Logger  # pytype: disable=module-attr
        logger.log_verbose("Cleaning socket and context.")
        socket_close_timeout_s = 0
        if self.socket:
            self.socket.close(linger=socket_close_timeout_s)
        if self.context:
            self.context.term()
        self.socket = None
        self.context = None

        logger.log_verbose("Removing allocated shared memory.")

        self.shm_request_manager.dispose()
        self.shm_response_manager.dispose()

        logger.log_verbose("Finalized.")

    @property
    def model_supports_batching(self) -> bool:
        """Return if model supports batching.

        Returns:
            True if model support batching, False otherwise.
        """
        return self.model_config["max_batch_size"] > 0

    def _get_instance_socket(self, shared_memory_socket):
        handshake_socket = self.context.socket(zmq.REQ)
        handshake_socket.connect(shared_memory_socket)
        handshake_socket.send_string("get_instance_socket")
        instance_shared_memory_socket = handshake_socket.recv().decode("utf-8")
        handshake_socket.close()
        return instance_shared_memory_socket

    def _exec_requests(self, inputs, logger):
        logger.log_verbose("Execute batch processing.")
        try:

            logger.log_verbose("Copying inputs to shared memory.")
            input_tensor_infos = self.shm_request_manager.to_shm(
                inputs, lambda data, req: MetaRequestResponse(data=data, parameters=req.parameters)
            )

            logger.log_verbose("Sending request to socket.")
            request = InferenceHandlerRequest(
                requests=input_tensor_infos, memory_name=self.shm_request_manager.memory_name()
            )
            self.socket.send(request.as_bytes())

            logger.log_verbose("Waiting for response.")
            response_payload = self.socket.recv()
            response = InferenceHandlerResponse.from_bytes(response_payload)

            logger.log_verbose(f"Response: {response.responses}")

            logger.log_verbose("Response obtained.")

            if response.error:
                raise pb_utils.TritonModelException(response.error)  # pytype: disable=module-attr

            logger.log_verbose("Preparing output arrays.")

            output_array = self.shm_response_manager.from_shm(
                response.responses, response.memory_name, lambda data, _resp: Response(data)
            )

            logger.log_verbose(f"Array: {output_array}")

        except pb_utils.TritonModelException:  # pytype: disable=module-attr
            raise
        except Exception:
            msg = traceback.format_exc()
            raise pb_utils.TritonModelException(f"Model execute error: {msg}")  # pytype: disable=module-attr

        return output_array
