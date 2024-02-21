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
import asyncio
import enum
import json
import random
import time
import typing

import numpy as np
import pytest

from pytriton.model_config.triton_model_config import TensorSpec
from pytriton.proxy.data import BaseRequestsResponsesSerializerDeserializer
from pytriton.proxy.types import Request, Requests, Response, Responses

TensorSpecs = typing.List[TensorSpec]
ParamsSpecs = typing.Dict[str, typing.Dict]


class _TestInferenceError(Exception):
    def __init__(self, message: str):
        self._message = message

    def __str__(self) -> str:
        return self._message

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _TestInferenceError) and self.message == other.message

    @property
    def message(self):
        return self._message


def _passthrough_response_from_request(request: Request) -> Response:
    return Response(data=request.data.copy())


class RandomRequestsGeneratorIterator:
    def __init__(
        self,
        inputs_specs: TensorSpecs,
        outputs_specs: TensorSpecs,
        parameters_specs: ParamsSpecs,
        n: int,
        max_batch_size: int,
    ):
        """Requests generator that generates random requests.

        Args:
            inputs_specs: inputs specs
            outputs_specs: outputs specs
            parameters_specs: parameters specs
            n: number of requests to generate
            max_batch_size: maximum batch size
        """
        self._inputs_specs = inputs_specs
        self._outputs_specs = outputs_specs
        self._parameters_specs = parameters_specs

        self._n = n
        self._max_batch_size = max_batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self._n == 0:
            raise StopIteration()

        # generate random number of ints which sum-up to self._max_batch_size
        batch_sizes = []
        while sum(batch_sizes) < self._max_batch_size:
            batch_sizes.append(random.randint(1, self._max_batch_size - sum(batch_sizes)))

        # to not always send requests filling max batch size
        batch_sizes = random.sample(batch_sizes, random.randint(1, len(batch_sizes)))

        # trim batch sizes that their sum not exceed self._n but keep at least one
        while sum(batch_sizes) > self._n and len(batch_sizes) > 1:
            batch_sizes.pop()
        if sum(batch_sizes) > self._n:
            batch_sizes[-1] -= sum(batch_sizes) - self._n

        self._n -= sum(batch_sizes)
        return [self._generate_request(batch_size) for batch_size in batch_sizes]

    def _generate_request(self, batch_size):
        inputs = {
            spec.name: np.random.rand(*((batch_size,) + spec.shape)).astype(spec.dtype) for spec in self._inputs_specs
        }
        parameters = {name: spec["type"](spec["default"]) for name, spec in self._parameters_specs.items()}
        return Request(data=inputs, parameters=parameters)


class RandomRequestsGenerator:
    def __init__(
        self,
        inputs_specs: TensorSpecs,
        outputs_specs: TensorSpecs,
        parameters_specs: ParamsSpecs,
        n: int,
        max_batch_size: int,
    ):
        """Requests generator that generates random requests.

        Args:
            inputs_specs: inputs specs
            outputs_specs: outputs specs
            parameters_specs: parameters specs
            n: number of requests to generate
            max_batch_size: maximum batch size
        """
        self._inputs_specs = inputs_specs
        self._outputs_specs = outputs_specs
        self._parameters_specs = parameters_specs
        self._n = n
        self._max_batch_size = max_batch_size

    def __iter__(self):
        return RandomRequestsGeneratorIterator(
            self._inputs_specs,
            self._outputs_specs,
            self._parameters_specs,
            self._n,
            self._max_batch_size,
        )


class TestAsyncSerializerDeserializer:
    async def serialize_requests(self, requests: Requests) -> bytes:
        def _request_as_dict(request):
            return {
                "data": {name: [array.tolist(), str(array.dtype)] for name, array in request.data.items()},
                "parameters": request.parameters,
            }

        return json.dumps([_request_as_dict(request) for request in requests]).encode("utf-8")

    async def deserialize_requests(self, requests_payload: bytes) -> Requests:
        def _request_from_dict(request_dict):
            return Request(
                data={name: np.array(value, dtype=dtype) for name, (value, dtype) in request_dict["data"].items()},
                parameters=request_dict["parameters"],
            )

        return [_request_from_dict(request_dict) for request_dict in json.loads(requests_payload.decode("utf-8"))]

    async def serialize_responses(self, responses: Responses) -> bytes:
        def _response_as_dict(response):
            return {"data": {name: [array.tolist(), str(array.dtype)] for name, array in response.data.items()}}

        return json.dumps([_response_as_dict(response) for response in responses]).encode("utf-8")

    async def deserialize_responses(self, responses_payload: bytes) -> Responses:
        def _response_from_dict(response_dict):
            return Response(
                data={name: np.array(value, dtype=dtype) for name, (value, dtype) in response_dict["data"].items()},
            )

        return [_response_from_dict(response_dict) for response_dict in json.loads(responses_payload.decode("utf-8"))]


class TestSyncSerializerDeserializer(BaseRequestsResponsesSerializerDeserializer):
    def serialize_requests(self, requests: Requests) -> bytes:
        def _request_as_dict(request):
            return {
                "data": {name: [array.tolist(), str(array.dtype)] for name, array in request.data.items()},
                "parameters": request.parameters,
            }

        return json.dumps([_request_as_dict(request) for request in requests]).encode("utf-8")

    def deserialize_requests(self, requests_payload: bytes) -> Requests:
        def _request_from_dict(request_dict):
            return Request(
                data={name: np.array(value, dtype=dtype) for name, (value, dtype) in request_dict["data"].items()},
                parameters=request_dict["parameters"],
            )

        return [_request_from_dict(request_dict) for request_dict in json.loads(requests_payload.decode("utf-8"))]

    def serialize_responses(self, responses: Responses) -> bytes:
        def _response_as_dict(response):
            return {"data": {name: [array.tolist(), str(array.dtype)] for name, array in response.data.items()}}

        return json.dumps([_response_as_dict(response) for response in responses]).encode("utf-8")

    def deserialize_responses(self, responses_payload: bytes) -> Responses:
        def _response_from_dict(response_dict):
            return Response(
                data={name: np.array(value, dtype=dtype) for name, (value, dtype) in response_dict["data"].items()},
            )

        return [_response_from_dict(response_dict) for response_dict in json.loads(responses_payload.decode("utf-8"))]

    def free_requests_resources(self, requests_payload: bytes):
        pass

    def free_responses_resources(self, responses_payload: bytes):
        pass


@pytest.fixture(scope="function")
def passthrough_requests_generator_factory():
    def _dummy_requests_generator(n, max_batch_size):
        requests_generator = RandomRequestsGenerator(
            inputs_specs=[TensorSpec(name="input1", shape=(2, 2), dtype=np.uint8)],
            outputs_specs=[TensorSpec(name="output1", shape=(2, 2), dtype=np.uint8)],
            parameters_specs={"param1": {"type": int, "default": 0}},
            n=n,
            max_batch_size=max_batch_size,
        )
        return requests_generator

    return _dummy_requests_generator


@pytest.fixture(scope="function")
def sleep_s():
    return max(0.0001, random.random() / 1000)  # sleep 0.1ms-1ms


class CallableType(enum.Enum):
    FUNCTION = "function"
    METHOD = "method"
    CALLABLE = "callable"
    FUNCTION_COROUTINE = "function_coroutine"
    METHOD_COROUTINE = "method_coroutine"
    CALLABLE_COROUTINE = "callable_coroutine"


def _passthrough_fn(requests: Requests, raise_error: bool, sleep_s: float) -> Responses:
    time.sleep(sleep_s)
    if raise_error:
        raise _TestInferenceError("Inference error")
    return [_passthrough_response_from_request(request) for request in requests]


def _streaming_passthrough_fn(
    requests: Requests,
    raise_error: bool,
    sleep_s: float,
) -> typing.Generator[Responses, None, None]:
    for _ in range(8):
        time.sleep(sleep_s)
        yield [_passthrough_response_from_request(request) for request in requests]
    if raise_error:
        raise _TestInferenceError("Inference error")


async def _passthrough_coro(requests: Requests, raise_error: bool, sleep_s: float) -> Responses:
    await asyncio.sleep(sleep_s)
    if raise_error:
        raise _TestInferenceError("Inference error")
    return [_passthrough_response_from_request(request) for request in requests]


async def _streaming_passthrough_coro(
    requests: Requests,
    raise_error: bool,
    sleep_s: float,
) -> typing.AsyncGenerator[Responses, None]:
    for _ in range(8):
        await asyncio.sleep(sleep_s)
        yield [_passthrough_response_from_request(request) for request in requests]
    if raise_error:
        raise _TestInferenceError("Inference error")


def _make_passthrough_function(streaming: bool, sleep_s: float, raise_error: bool = False):
    sleep_s = sleep_s if sleep_s is not None else max(0.0001, random.random() / 1000)  # sleep 0.1ms-1ms

    def _passthrough_fn_wrapper(requests: Requests) -> Responses:
        return _passthrough_fn(requests, raise_error=raise_error, sleep_s=sleep_s)

    def _streaming_passthrough_fn_wrapper(requests: Requests) -> typing.Generator[Responses, None, None]:
        yield from _streaming_passthrough_fn(requests, raise_error=raise_error, sleep_s=sleep_s)

    return _streaming_passthrough_fn_wrapper if streaming else _passthrough_fn_wrapper


def _make_passthrough_method(streaming: bool, sleep_s: float, raise_error: bool = False):
    class _PassthroughCallable:
        def infer(self, requests: Requests) -> Responses:
            return _passthrough_fn(requests, raise_error=raise_error, sleep_s=sleep_s)

    class _StreamingPassthroughCallable:
        def infer(self, requests: Requests) -> typing.Generator[Responses, None, None]:
            yield from _streaming_passthrough_fn(requests, raise_error=raise_error, sleep_s=sleep_s)

    return _StreamingPassthroughCallable().infer if streaming else _PassthroughCallable().infer


def _make_passthrough_callable(streaming: bool, sleep_s: float, raise_error: bool = False):
    class _PassthroughCallable:
        def __call__(self, requests: Requests) -> Responses:
            return _passthrough_fn(requests, raise_error=raise_error, sleep_s=sleep_s)

    class _StreamingPassthroughCallable:
        def __call__(self, requests: Requests) -> typing.Generator[Responses, None, None]:
            yield from _streaming_passthrough_fn(requests, raise_error=raise_error, sleep_s=sleep_s)

    return _StreamingPassthroughCallable() if streaming else _PassthroughCallable()


def _make_passthrough_coro(streaming: bool, sleep_s: float, raise_error: bool = False):
    async def _passthrough_coro_wrapper(requests: Requests) -> Responses:
        return await _passthrough_coro(requests, raise_error=raise_error, sleep_s=sleep_s)

    async def _streaming_passthrough_coro_wrapper(requests: Requests) -> typing.AsyncGenerator[Responses, None]:
        async for responses in _streaming_passthrough_coro(requests, raise_error=raise_error, sleep_s=sleep_s):
            yield responses

    return _streaming_passthrough_coro_wrapper if streaming else _passthrough_coro_wrapper


def _make_passthrough_method_coro(streaming: bool, sleep_s: float, raise_error: bool = False):
    class _PassthroughCallableCoro:
        async def infer(self, requests: Requests) -> Responses:
            return await _passthrough_coro(requests, raise_error=raise_error, sleep_s=sleep_s)

    class _StreamingPassthroughCallableCoro:
        async def infer(self, requests: Requests) -> typing.AsyncGenerator[Responses, None]:
            async for responses in _streaming_passthrough_coro(requests, raise_error=raise_error, sleep_s=sleep_s):
                yield responses

    return _StreamingPassthroughCallableCoro().infer if streaming else _PassthroughCallableCoro().infer


def _make_passthrough_callable_coro(streaming: bool, sleep_s: float, raise_error: bool = False):
    class _PassthroughCallableCoro:
        async def __call__(self, requests: Requests) -> Responses:
            return await _passthrough_coro(requests, raise_error=raise_error, sleep_s=sleep_s)

    class _StreamingPassthroughCallableCoro:
        async def __call__(self, requests: Requests) -> typing.AsyncGenerator[Responses, None]:
            async for responses in _streaming_passthrough_coro(requests, raise_error=raise_error, sleep_s=sleep_s):
                yield responses

    return _StreamingPassthroughCallableCoro() if streaming else _PassthroughCallableCoro()


@pytest.fixture(scope="function")
def make_passthrough_callable():
    def __make_passthrough_callable(
        callable_type: CallableType,
        streaming: bool = False,
        sleep_s: typing.Optional[float] = None,
        raise_error: bool = False,
    ):
        sleep_s = sleep_s if sleep_s is not None else max(0.0001, random.random() / 1000)  # sleep 0.1ms-1ms

        return {
            CallableType.FUNCTION: _make_passthrough_function,
            CallableType.METHOD: _make_passthrough_method,
            CallableType.CALLABLE: _make_passthrough_callable,
            CallableType.FUNCTION_COROUTINE: _make_passthrough_coro,
            CallableType.METHOD_COROUTINE: _make_passthrough_method_coro,
            CallableType.CALLABLE_COROUTINE: _make_passthrough_callable_coro,
        }[callable_type](streaming=streaming, sleep_s=sleep_s, raise_error=raise_error)

    return __make_passthrough_callable
