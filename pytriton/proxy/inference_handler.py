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
import logging
import threading as th
import traceback
import typing
from typing import Callable

import numpy as np
import zmq  # pytype: disable=import-error

from pytriton.exceptions import PyTritonUnrecoverableError
from pytriton.model_config.triton_model_config import TritonModelConfig
from pytriton.proxy.communication import Request, Response, ShmManager

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
        zmq_context: zmq.Context,
    ):
        """Create a PythonBackend object.

        Args:
            model_callable: A model callable to pass and receive the data
            model_config: Triton model configuration
            shared_memory_socket: Socket path for shared memory communication
            zmq_context: zero mq context
        """
        super().__init__()
        self._model_config = model_config
        self._model_callable = model_callable
        self.stopped = False

        self.shm_request_manager = ShmManager()
        self.shm_response_manager = ShmManager()

        self.zmq_context = zmq_context
        self.socket = None
        self.shared_memory_socket = shared_memory_socket

        self._proxy_backend_events_observers: typing.List[InferenceEventsHandler] = []

    def run(self) -> None:
        """Start the InferenceHandler communication."""
        self.socket = self.zmq_context.socket(zmq.REP)
        try:
            LOGGER.debug(f"Binding IPC socket at {self.shared_memory_socket}.")
            self.socket.bind(self.shared_memory_socket)

            while not self.stopped:
                LOGGER.debug(f"Waiting for requests from proxy model for {self._model_config.model_name}.")
                request_payload = self.socket.recv()
                request = Request.from_bytes(request_payload)

                LOGGER.debug(f"Preparing inputs for {self._model_config.model_name}.")
                inputs = self.shm_request_manager.from_shm(request.inputs, request.memory_name)

                try:
                    LOGGER.debug(f"Processing inference callback for {self._model_config.model_name}.")
                    outputs = self._model_callable(inputs)

                    LOGGER.debug(f"Validating outputs for {self._model_config.model_name}.")
                    self._validate_outputs(outputs)

                    output_tensor_infos = self.shm_response_manager.to_shm(outputs)

                    response = Response(
                        outputs=output_tensor_infos, memory_name=self.shm_response_manager.memory_name()
                    )
                except PyTritonUnrecoverableError:
                    error = traceback.format_exc()
                    response = Response(error=error)
                    LOGGER.error(
                        "Unrecoverable error thrown during calling model callable. "
                        "Shutting down Triton Inference Server. "
                        f"{error}"
                    )
                    self.stopped = True
                    self._notify_proxy_backend_observers(InferenceHandlerEvent.UNRECOVERABLE_ERROR, error)
                except Exception:
                    error = traceback.format_exc()
                    response = Response(error=error)
                    LOGGER.error(f"Error occurred during calling model callable: {error}")

                LOGGER.debug(f"Send response to proxy model for {self._model_config.model_name}.")
                self.socket.send(response.as_bytes())
        except zmq.error.ContextTerminated:
            LOGGER.info("Context was terminated. InferenceHandler will be closed.")
        except Exception as exception:
            LOGGER.error("Internal proxy backend error. InferenceHandler will be closed.")
            LOGGER.exception(exception)
        finally:
            LOGGER.info("Closing socket")
            socket_close_timeout_s = 0
            self.socket.close(linger=socket_close_timeout_s)
            LOGGER.info("Closing buffers")
            self.shm_request_manager.dispose()
            self.shm_response_manager.dispose()

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
