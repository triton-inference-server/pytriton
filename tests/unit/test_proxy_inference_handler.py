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
import logging

import pytest

from pytriton.proxy.inference import InferenceHandler, PyTritonResponseFlags, _AsyncGenForCallableAdapter
from tests.unit.conftest import CallableType, _TestInferenceError
from tests.unit.utils import check_all_expected_calls_made, verify_equalness_of_dicts_with_ndarray

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


INFERENCE_CALLABLE_TYPES = [
    CallableType.FUNCTION,
    CallableType.CALLABLE,
    CallableType.COROUTINE,
    CallableType.CALLABLE_COROUTINE,
]


@pytest.mark.parametrize("callable_type", INFERENCE_CALLABLE_TYPES)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("raise_error", [False])
async def test_async_gen_for_callable_adapter(
    passthrough_requests_generator_factory,
    make_passthrough_callable,
    callable_type,
    streaming,
    raise_error,
):
    infer_callable = make_passthrough_callable(callable_type, streaming=streaming, raise_error=raise_error)
    async_gen = _AsyncGenForCallableAdapter(infer_callable)
    requests_generator = passthrough_requests_generator_factory(n=8, max_batch_size=32)
    for requests in requests_generator:
        async for responses in async_gen(requests):
            assert len(requests) == len(responses)
            for request, response in zip(requests, responses):
                verify_equalness_of_dicts_with_ndarray(request, response)


@pytest.mark.parametrize("callable_type", INFERENCE_CALLABLE_TYPES)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("raise_error", [True])
async def test_async_gen_for_callable_adapter_passes_error(
    passthrough_requests_generator_factory,
    make_passthrough_callable,
    callable_type,
    streaming,
    raise_error,
):
    infer_callable = make_passthrough_callable(callable_type, streaming=streaming, raise_error=raise_error)
    async_gen = _AsyncGenForCallableAdapter(infer_callable)
    requests_generator = passthrough_requests_generator_factory(n=8, max_batch_size=32)

    with pytest.raises(_TestInferenceError, match="Inference error"):
        for requests in requests_generator:
            async for responses in async_gen(requests):
                assert len(requests) == len(responses)
                for request, response in zip(requests, responses):
                    verify_equalness_of_dicts_with_ndarray(request, response)


@pytest.mark.parametrize("callable_type", INFERENCE_CALLABLE_TYPES)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("raise_error", [False])
def test_inference_handler_fast_start_stop(
    make_passthrough_callable,
    callable_type,
    streaming,
    raise_error,
    mocker,
):
    infer_callable = make_passthrough_callable(callable_type, streaming=streaming, raise_error=raise_error)

    requests_responses_connector = mocker.Mock()
    validator = mocker.Mock()
    inference_handler = None
    try:
        inference_handler = InferenceHandler(infer_callable, requests_responses_connector, validator)
        inference_handler.start()
    finally:
        if inference_handler is not None:
            inference_handler.stop()
            inference_handler.join(timeout=5)
            assert not inference_handler.is_alive()

            requests_responses_connector.register_inference_hook.assert_called_once_with(
                inference_handler.run_inference
            )
            requests_responses_connector.unregister_inference_hook.assert_called_once_with(
                inference_handler.run_inference
            )


@pytest.mark.parametrize("callable_type", INFERENCE_CALLABLE_TYPES)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("raise_error", [False, True])
async def test_inference_handler_inference(
    passthrough_requests_generator_factory,
    make_passthrough_callable,
    callable_type,
    streaming,
    raise_error,
    mocker,
):
    infer_callable = make_passthrough_callable(callable_type, streaming=streaming, raise_error=raise_error)
    requests_iterator = iter(passthrough_requests_generator_factory(n=8, max_batch_size=32))

    requests_responses_connector = mocker.Mock()
    validator = mocker.Mock()
    inference_handler = None

    expected_connector_calls = []
    futures_to_check = []
    try:
        inference_handler = InferenceHandler(infer_callable, requests_responses_connector, validator)
        inference_handler.start()
        expected_connector_calls.append(mocker.call.register_inference_hook(inference_handler.run_inference))

        idx = 0
        while True:
            try:
                scope = {"requests_id": idx.to_bytes(4, "big")}
                requests = next(requests_iterator)
                inference_future = inference_handler.run_inference(scope, requests)
                futures_to_check.append(inference_future)

                async_gen = _AsyncGenForCallableAdapter(infer_callable)
                try:
                    async for expected_responses in async_gen(requests):
                        expected_connector_calls.append(mocker.call.send(scope, 0, expected_responses))
                    expected_connector_calls.append(mocker.call.send(scope, PyTritonResponseFlags.EOS, None))
                except _TestInferenceError as e:
                    expected_connector_calls.append(
                        mocker.call.send(scope, PyTritonResponseFlags.EOS | PyTritonResponseFlags.ERROR, e)
                    )

                idx += 1
            except StopIteration:
                break

    finally:
        if inference_handler is not None:
            inference_handler.stop()  # all inference requests done till now should be finished
            expected_connector_calls.append(mocker.call.unregister_inference_hook(inference_handler.run_inference))
            inference_handler.join(timeout=5)

            assert not inference_handler.is_alive()
            assert all(inference_future.result() is None for inference_future in futures_to_check)

            # ignore order, as we don't know which inference calls coroutines
            # are run before stop() is called and which are run after
            check_all_expected_calls_made(
                expected_connector_calls, list(requests_responses_connector.mock_calls), any_order=True
            )


@pytest.mark.parametrize("callable_type", INFERENCE_CALLABLE_TYPES)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("raise_error", [False])
async def test_inference_handler_propagates_validaton_error(
    passthrough_requests_generator_factory,
    make_passthrough_callable,
    callable_type,
    streaming,
    raise_error,
    mocker,
):
    infer_callable = make_passthrough_callable(callable_type, streaming=streaming, raise_error=raise_error)
    requests_iterator = iter(passthrough_requests_generator_factory(n=8, max_batch_size=32))

    requests_responses_connector = mocker.Mock()

    validator = mocker.Mock()
    validator.validate_responses.side_effect = _TestInferenceError("Validation error")

    inference_handler = None

    expected_connector_calls = []
    expected_validator_calls = []
    futures_to_check = []
    try:
        inference_handler = InferenceHandler(infer_callable, requests_responses_connector, validator)
        inference_handler.start()
        expected_connector_calls.append(mocker.call.register_inference_hook(inference_handler.run_inference))

        idx = 0
        while True:
            try:
                scope = {"requests_id": idx.to_bytes(4, "big")}
                requests = next(requests_iterator)
                inference_future = inference_handler.run_inference(scope, requests)
                futures_to_check.append(inference_future)

                async_gen = _AsyncGenForCallableAdapter(infer_callable)
                async for expected_responses in async_gen(requests):
                    expected_validator_calls.append(mocker.call.validate_responses(requests, expected_responses))
                    break

                expected_connector_calls.append(
                    mocker.call.send(
                        scope,
                        PyTritonResponseFlags.EOS | PyTritonResponseFlags.ERROR,
                        _TestInferenceError("Validation error"),
                    )
                )

                idx += 1
            except StopIteration:
                break

    finally:
        if inference_handler is not None:
            inference_handler.stop()  # all inference requests done till now should be finished
            expected_connector_calls.append(mocker.call.unregister_inference_hook(inference_handler.run_inference))
            inference_handler.join(timeout=5)

            assert not inference_handler.is_alive()
            assert all(inference_future.result() is None for inference_future in futures_to_check)

            check_all_expected_calls_made(expected_validator_calls, list(validator.mock_calls))

            # ignore order, as we don't know which inference calls coroutines
            # are run before stop() is called and which are run after
            check_all_expected_calls_made(
                expected_connector_calls, list(requests_responses_connector.mock_calls), any_order=True
            )


# cancel scheduled tasks
