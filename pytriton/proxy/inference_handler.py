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
"""InferenceHandler class.

Create a connection between the Python and Triton Inference Server to pass request and response between the
python model deployed on Triton and model in current environment.

    Examples of use:

        model_config = ModelConfigParser.from_file("repo/model/config.pbtxt")
        handler = InferenceHandler(
            model_callable=infer_func,
            model_config=model_config,
            shared_memory_socket="ipc:///tmp/shared-memory-socker.ipc"
        )
        handler.run()

"""
import enum
import itertools
import logging
import threading as th
import traceback
import typing
from typing import Callable

import numpy as np
import zmq  # pytype: disable=import-error

from pytriton.exceptions import PyTritonUnrecoverableError
from pytriton.model_config.triton_model_config import TritonModelConfig
from pytriton.proxy.communication import (
    InferenceHandlerRequests,
    InferenceHandlerResponses,
    MetaRequestResponse,
    TensorStore,
)
from pytriton.proxy.types import Request

LOGGER = logging.getLogger(__name__)


class InferenceHandlerEvent(enum.Enum):
    """Represents proxy backend event."""

    STARTED = "started"
    UNRECOVERABLE_ERROR = "unrecoverable-error"
    FINISHED = "finished"


InferenceEventsHandler = typing.Callable[["InferenceHandler", InferenceHandlerEvent, typing.Optional[typing.Any]], None]


class InferenceHandler(th.Thread):
    """InferenceHandler class for handling the communication between Triton and local model."""

    def __init__(
        self,
        model_callable: Callable,
        model_config: TritonModelConfig,
        shared_memory_socket: str,
        data_store_socket: str,
        zmq_context: zmq.Context,
    ):
        """Create a PythonBackend object.

        Args:
            model_callable: A model callable to pass and receive the data
            model_config: Triton model configuration
            shared_memory_socket: Socket path for shared memory communication
            data_store_socket: Socket path for data store communication
            zmq_context: zero mq context
        """
        super().__init__()
        self._model_config = model_config
        self._model_callable = model_callable
        self.stopped = False

        self._tensor_store = TensorStore(data_store_socket)

        self.zmq_context = zmq_context
        self.socket = None
        self.shared_memory_socket = shared_memory_socket

        self._proxy_backend_events_observers: typing.List[InferenceEventsHandler] = []

    def run(self) -> None:
        """Start the InferenceHandler communication."""
        self.socket = self.zmq_context.socket(zmq.REP)
        model_name = self._model_config.model_name
        try:
            LOGGER.debug(f"Binding IPC socket at {self.shared_memory_socket}.")
            self.socket.bind(self.shared_memory_socket)
            self._tensor_store.connect()

            while not self.stopped:
                LOGGER.debug(f"Waiting for requests from proxy model for {model_name}.")
                request_payload = self.socket.recv()
                requests = InferenceHandlerRequests.from_bytes(request_payload).requests

                LOGGER.debug(f"Preparing inputs for {model_name}.")
                inputs = [
                    Request(
                        data={
                            input_name: self._tensor_store.get(tensor_id)
                            for input_name, tensor_id in request.data.items()
                        },
                        parameters=request.parameters,
                    )
                    for request in requests
                ]

                try:
                    LOGGER.debug(f"Processing inference callback for {model_name}.")
                    responses = self._model_callable(inputs)

                    LOGGER.debug(f"Validating outputs for {model_name}.")
                    self._validate_outputs(responses)

                    # to avoid reallocation need to declare required size of the buffer here
                    LOGGER.debug(f"Copying outputs to shared memory for {model_name}.")
                    for tensor_id in itertools.chain(*[request.data.values() for request in requests]):
                        self._tensor_store.release_block(tensor_id)

                    output_arrays_with_coords = [
                        (response_idx, output_name, tensor)
                        for response_idx, response in enumerate(responses)
                        for output_name, tensor in response.items()
                    ]
                    tensor_ids = self._tensor_store.put([tensor for _, _, tensor in output_arrays_with_coords])
                    responses = [{} for _ in range(len(responses))]
                    for (response_idx, output_name, _), tensor_id in zip(output_arrays_with_coords, tensor_ids):
                        responses[response_idx][output_name] = tensor_id

                    responses = InferenceHandlerResponses(
                        responses=[MetaRequestResponse(data=response) for response in responses]
                    )
                except PyTritonUnrecoverableError:
                    error = traceback.format_exc()
                    responses = InferenceHandlerResponses(error=error)
                    LOGGER.error(
                        "Unrecoverable error thrown during calling model callable. "
                        "Shutting down Triton Inference Server. "
                        f"{error}"
                    )
                    self.stopped = True
                    self._notify_proxy_backend_observers(InferenceHandlerEvent.UNRECOVERABLE_ERROR, error)
                except Exception:
                    error = traceback.format_exc()
                    responses = InferenceHandlerResponses(error=error)
                    LOGGER.error(f"Error occurred during calling model callable: {error}")

                LOGGER.debug(f"Send response to proxy model for {model_name}.")
                self.socket.send(responses.as_bytes())
        except zmq.error.ContextTerminated:
            LOGGER.info("Context was terminated. InferenceHandler will be closed.")
        except Exception as exception:
            LOGGER.error("Internal proxy backend error. InferenceHandler will be closed.")
            LOGGER.exception(exception)
        finally:
            LOGGER.info("Closing socket")
            socket_close_timeout_s = 0
            self.socket.close(linger=socket_close_timeout_s)
            LOGGER.info("Closing TensorStore")
            self._tensor_store.close()

            LOGGER.info("Leaving proxy backend thread")
            self._notify_proxy_backend_observers(InferenceHandlerEvent.FINISHED, None)

    def _validate_outputs(self, outputs):
        if not isinstance(outputs, list):
            raise ValueError("Outputs returned by model callable must be list of request dicts with numpy arrays")

        for request in outputs:
            if not isinstance(request, dict):
                raise ValueError("Outputs returned by model callable must be list of request dicts with numpy arrays")
            for key, value in request.items():
                if not isinstance(key, str):
                    raise ValueError("Not all keys returned by model callable are string")
                if not isinstance(value, np.ndarray):
                    raise ValueError("Not all values returned by model callable are numpy arrays")
                else:
                    allowed_kind = "biufOSU"
                    if value.dtype.kind not in allowed_kind:
                        raise ValueError(
                            f"Only bool, numeric, string, unicode and object arrays are supported by Triton (dtype.kind: {allowed_kind}). "
                            f"Returned {key} has {value.dtype.kind} dtype.kind."
                        )
                    if value.dtype.kind == "O":
                        if isinstance(value.item(0), str):
                            raise ValueError(
                                "Use string/byte-string instead of object for passing string in NumPy array."
                            )
                        elif not isinstance(value.item(0), bytes):
                            raise ValueError(
                                f"Only bytes as objects dtype are supported by PyTriton. Returned {key} has {type(value.item(0))} type."
                            )

    def stop(self) -> None:
        """Stop the InferenceHandler communication."""
        LOGGER.info("Closing proxy")
        self.stopped = True
        self.join()

    def on_proxy_backend_event(self, proxy_backend_event_handle_fn: InferenceEventsHandler):
        """Register InferenceEventsHandler callable.

        Args:
            proxy_backend_event_handle_fn: function to be called when proxy backend events arises
        """
        self._proxy_backend_events_observers.append(proxy_backend_event_handle_fn)

    def _notify_proxy_backend_observers(self, event: InferenceHandlerEvent, context: typing.Optional[typing.Any]):
        for proxy_backend_event_handle_fn in self._proxy_backend_events_observers:
            proxy_backend_event_handle_fn(self, event, context)
