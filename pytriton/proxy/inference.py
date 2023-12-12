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
"""Module with classes for executing received requests on model callables and sending responses back to Triton."""

import abc
import asyncio
import concurrent.futures
import enum
import inspect
import logging
import threading
import time
import traceback
import typing

from pytriton.exceptions import PyTritonUnrecoverableError
from pytriton.proxy.communication import PyTritonResponseFlags, RequestsServerClient
from pytriton.proxy.types import Requests, Responses, ResponsesNoneOrError, Scope
from pytriton.proxy.validators import TritonResultsValidator

LOGGER = logging.getLogger(__name__)


class InferenceHandlerEvent(enum.Enum):
    """Represents proxy backend event."""

    STARTED = "started"
    CLOSING = "closing"
    CLOSED = "closed"
    UNRECOVERABLE_ERROR = "unrecoverable-error"


InferenceHandlerEventsHandler = typing.Callable[
    ["InferenceHandler", InferenceHandlerEvent, typing.Optional[typing.Any]], None
]


class _AsyncGenForCallableAdapter:
    """Adapter for converting a callable to an async generator."""

    def __new__(cls, inference_callable):
        """Create an async generator from a callable.

        Args:
            inference_callable: A callable to convert to an async generator.
        """
        if callable(inference_callable) and not inspect.isfunction(inference_callable):
            inference_callable = inference_callable.__call__

        if inspect.isasyncgenfunction(inference_callable):
            return inference_callable
        elif inspect.iscoroutinefunction(inference_callable):

            async def _callable(requests):
                yield await inference_callable(requests)

            return _callable
        elif inspect.isgeneratorfunction(inference_callable):

            async def _callable(requests):
                for result in inference_callable(requests):
                    yield result
                    await asyncio.sleep(0)

            return _callable
        else:

            async def _callable(requests):
                yield inference_callable(requests)
                await asyncio.sleep(0)

            return _callable


class BaseRequestsResponsesConnector(abc.ABC):
    """Base class for requests responses connector."""

    @abc.abstractmethod
    def register_inference_hook(self, run_inference_fn: typing.Callable[[Scope, Requests], concurrent.futures.Future]):
        """Register inference hook.

        Args:
            run_inference_fn: function to run inference on requests
        """
        pass

    @abc.abstractmethod
    def unregister_inference_hook(
        self, run_inference_fn: typing.Callable[[Scope, Requests], concurrent.futures.Future]
    ):
        """Unregister inference hook.

        Args:
            run_inference_fn: function to run inference on requests
        """
        pass

    @abc.abstractmethod
    def send(self, scope: Scope, flags: PyTritonResponseFlags, responses: ResponsesNoneOrError):
        """Send responses back to server.

        Args:
            scope: scope of the requests
            flags: flags for responses
            responses: responses to send back to server
        """
        pass


class RequestsResponsesConnector(threading.Thread, BaseRequestsResponsesConnector):
    """Thread for handling requests received from Triton."""

    INFERENCE_FN_REGISTER_WAIT_TIME_S = 5

    def __init__(self, url: str, serializer_deserializer):
        """Requests Server Client thread.

        Args:
            url: url of the requests server
            serializer_deserializer: serializer and deserializer for requests and responses
        """
        self._requests_server_client = RequestsServerClient(url, self.handle_requests)
        self._serializer_deserializer = serializer_deserializer

        self._responses_queues: typing.Dict[bytes, asyncio.Queue] = {}
        self._run_inference_fn: typing.Optional[typing.Callable[[Scope, Requests], concurrent.futures.Future]] = None
        self._run_inference_condition = threading.Condition()

        super().__init__(daemon=True, name=f"{self._requests_server_client._name}-comm_thread")

    def run(self):
        """Requests Server Client thread run method."""
        self._requests_server_client.run()

    def close(self):
        """Close Requests Server Client thread."""
        self._requests_server_client.shutdown()

    async def handle_requests(self, scope, requests_payload: bytes, send):
        """Handle requests received from Triton.

        Args:
            scope: scope of the requests
            requests_payload: requests payload to handle
            send: function to send responses back to Triton

        Returns:
            None

        Raises:
            Exception: if an error occurs while handling requests
        """
        requests_id = scope["requests_id"]
        queue = self._responses_queues[requests_id] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _wait_for_inference_fn(timeout_s: float):
            with self._run_inference_condition:
                return self._run_inference_condition.wait_for(
                    lambda: self._run_inference_fn is not None, timeout=timeout_s
                )

        try:
            requests = await self.preprocess(scope, requests_payload)

            if self._run_inference_fn is None:
                await loop.run_in_executor(None, _wait_for_inference_fn, self.INFERENCE_FN_REGISTER_WAIT_TIME_S)

            with self._run_inference_condition:
                if self._run_inference_fn is None:
                    raise RuntimeError("Inference callable is not registered (inference handler is stopped)")

                run_inference_future = self._run_inference_fn(scope, requests)
            while True:
                (flags, responses_or_error) = await queue.get()
                if flags & PyTritonResponseFlags.ERROR:
                    error_msg = "".join(
                        traceback.format_exception(None, responses_or_error, responses_or_error.__traceback__)
                    )
                    error_msg = error_msg.encode()
                    await send(scope, flags, error_msg)
                    break

                responses_payload = await self.postprocess(scope, responses_or_error)

                await send(scope, flags, responses_payload)
                if flags & PyTritonResponseFlags.EOS:
                    break

            run_inference_future.result()
        except Exception:
            error_msg = traceback.format_exc().encode()
            flags = PyTritonResponseFlags.ERROR | PyTritonResponseFlags.EOS
            await send(scope, flags, error_msg)
        finally:
            await loop.run_in_executor(None, self._serializer_deserializer.free_requests_resources, requests_payload)
            self._responses_queues.pop(requests_id)
            LOGGER.debug(f"Finished handling requests for {scope['requests_id'].hex()}")

    async def preprocess(self, scope: Scope, requests_payload: bytes) -> Requests:
        """Preprocess requests before running inference on them.

        Currently, this method only deserializes requests.

        Args:
            scope: scope of the requests
            requests_payload: requests payload to preprocess

        Returns:
            deserialized requests
        """
        LOGGER.debug(f"Preprocessing requests for {scope['requests_id'].hex()}")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._serializer_deserializer.deserialize_requests, requests_payload)

    async def postprocess(self, scope: Scope, responses: Responses) -> bytes:
        """Postprocess responses before sending them back to Triton.

        Currently, this method only serializes responses.

        Args:
            scope: scope of the requests
            responses: responses to postprocess

        Returns:
            serialized responses
        """
        LOGGER.debug(f"Postprocessing responses for {scope['requests_id'].hex()}")
        if responses is None:
            return b""
        else:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._serializer_deserializer.serialize_responses, responses)

    def register_inference_hook(self, run_inference_fn: typing.Callable[[Scope, Requests], concurrent.futures.Future]):
        """Register inference hook.

        Args:
            run_inference_fn: function to run inference on requests
        """
        with self._run_inference_condition:
            self._run_inference_fn = run_inference_fn
            self._run_inference_condition.notify_all()

    def unregister_inference_hook(
        self, run_inference_fn: typing.Callable[[Scope, Requests], concurrent.futures.Future]
    ):
        """Unregister inference hook.

        Args:
            run_inference_fn: function to run inference on requests
        """
        with self._run_inference_condition:
            self._run_inference_fn = None
            self._run_inference_condition.notify_all()

    def send(self, scope: Scope, flags: PyTritonResponseFlags, responses: ResponsesNoneOrError):
        """Send responses back to server.

        Args:
            scope: scope of the requests
            flags: flags for responses
            responses: responses to send back to server
        """
        requests_id = scope["requests_id"]
        LOGGER.debug(f"Pushing responses for {scope['requests_id'].hex()} into responses queue ({flags}, {responses})")
        queue = self._responses_queues[requests_id]
        loop = self._requests_server_client.loop
        # use no_wait as there is no limit for responses queues
        loop.call_soon_threadsafe(queue.put_nowait, (flags, responses))


class InferenceHandler(threading.Thread):
    """Thread for running inference on requests."""

    def __init__(
        self,
        model_callable: typing.Callable,
        requests_responses_connector: BaseRequestsResponsesConnector,
        validator: TritonResultsValidator,
        name: typing.Optional[str] = None,
    ):
        """Inference Handler thread.

        Args:
            model_callable: model callable to run inference on requests
            requests_responses_connector: requests responses connector
            validator: validator for requests and responses
            name: name of the thread for easy of debugging
        """
        self._model_callable = _AsyncGenForCallableAdapter(model_callable)
        self._requests_responses_connector = requests_responses_connector
        self._validator = validator

        self._loop = None
        self._loop_condition = threading.Condition()
        self._inference_handler_events_observers: typing.List[InferenceHandlerEventsHandler] = []
        self._wait_for_schechuled_tasks_timeout_s = 20.0

        name = name or "inference_handler"
        super().__init__(daemon=True, name=name)

    def run(self):
        """Inference Handler thread run method."""
        with self._loop_condition:
            self._loop = asyncio.new_event_loop()
            self._loop_condition.notify_all()

        asyncio.set_event_loop(self._loop)

        try:
            self._notify_inference_handler_events_observers(InferenceHandlerEvent.STARTED, None)
            self._requests_responses_connector.register_inference_hook(self.run_inference)
            self._loop.run_forever()
        finally:
            self._notify_inference_handler_events_observers(InferenceHandlerEvent.CLOSING, None)
            try:
                _cancel_all_tasks(self._loop)
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                self._loop.run_until_complete(self._loop.shutdown_default_executor())
            finally:
                asyncio.set_event_loop(None)
                self._loop.close()

            self._notify_inference_handler_events_observers(InferenceHandlerEvent.CLOSED, None)

    def start(self):
        """Start Inference Handler."""
        super().start()
        with self._loop_condition:
            small_timeout_s = 5
            self._loop_condition.wait_for(lambda: self._loop is not None, timeout=small_timeout_s)

    def stop(self) -> None:
        """Stop Inference Handler."""
        LOGGER.info("Closing Inference Handler")
        self._requests_responses_connector.unregister_inference_hook(self.run_inference)
        if self._loop is not None:
            try:
                _wait_for_scheduled_tasks(
                    self._loop, self._handle_requests.__name__, timeout_s=self._wait_for_schechuled_tasks_timeout_s
                )
            except TimeoutError:
                LOGGER.warning(
                    "Timeout while waiting for submitted inference tasks to finish. Cancelling remaining tasks."
                )
            self._loop.call_soon_threadsafe(self._loop.stop)

    def on_inference_handler_event(self, inference_handler_events_handle_fn: InferenceHandlerEventsHandler):
        """Register InferenceHandlerEventsHandler callable.

        Args:
            inference_handler_events_handle_fn: function to be called when inference handler event arises
        """
        self._inference_handler_events_observers.append(inference_handler_events_handle_fn)

    def run_inference(self, scope: Scope, requests: Requests):
        """Run inference on requests.

        Args:
            scope: scope of the requests
            requests: requests to run inference on

        Returns:
            Future of inference task
        """
        return asyncio.run_coroutine_threadsafe(self._handle_requests(scope, requests), self._loop)

    async def _handle_requests(self, scope: Scope, requests: Requests):
        requests_id = scope["requests_id"]
        LOGGER.debug(f"Performing inference on requests={requests_id.hex()}")

        responses = None
        try:
            async for responses in self._model_callable(requests):
                self._validator.validate_responses(requests, responses)
                self._requests_responses_connector.send(scope, PyTritonResponseFlags(0), responses)
            self._requests_responses_connector.send(scope, PyTritonResponseFlags.EOS, None)
        except (Exception, asyncio.CancelledError) as e:
            error_msg = traceback.format_exc()
            if isinstance(e, PyTritonUnrecoverableError):
                LOGGER.error(
                    f"Unrecoverable error thrown during handling requests={requests_id}. "
                    "Shutting down Triton Inference Server. "
                    f"{error_msg}"
                )
                self._notify_inference_handler_events_observers(InferenceHandlerEvent.UNRECOVERABLE_ERROR, error_msg)
                self.stop()
            else:
                LOGGER.warning(f"Exception while performing inference on requests={requests_id.hex()}: {error_msg}")
            self._requests_responses_connector.send(scope, PyTritonResponseFlags.ERROR | PyTritonResponseFlags.EOS, e)

        LOGGER.debug(f"Finished inference on requests={requests_id.hex()}")

    def _notify_inference_handler_events_observers(
        self,
        event: InferenceHandlerEvent,
        context: typing.Optional[typing.Any],
    ):
        for inference_handler_events_handler_fn in self._inference_handler_events_observers:
            inference_handler_events_handler_fn(self, event, context)


def _wait_for_scheduled_tasks(loop, coro_name, timeout_s: float):
    def _get_inference_tasks():
        # async generators are separate tasks, and have no names. we should wait for them to finish as well
        result = [
            task for task in asyncio.all_tasks(loop) if getattr(task.get_coro(), "__name__", None) in [coro_name, None]
        ]
        return result

    check_interval_s = 0.1
    while _get_inference_tasks():
        time.sleep(check_interval_s)
        timeout_s -= check_interval_s
        if timeout_s <= 0:
            raise TimeoutError(f"Timeout while waiting for {coro_name} tasks to finish")


def _cancel_all_tasks(loop):
    """From Python 3.8 asyncio/runners.py."""
    to_cancel = asyncio.all_tasks(loop)
    if not to_cancel:
        return

    for task in to_cancel:
        task.cancel()

    loop.run_until_complete(asyncio.gather(*to_cancel, return_exceptions=True))

    for task in to_cancel:
        if task.cancelled():
            continue
        if task.exception() is not None:
            loop.call_exception_handler(
                {
                    "message": "unhandled exception during asyncio.run() shutdown",
                    "exception": task.exception(),
                    "task": task,
                }
            )
