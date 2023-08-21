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
import dataclasses
import functools
import json
import logging
import time

import numpy as np
import pytest
import zmq

from pytriton.exceptions import PyTritonRuntimeError
from pytriton.model_config.triton_model_config import TensorSpec, TritonModelConfig
from pytriton.proxy.communication import InferenceHandlerRequests, MetaRequestResponse, TensorStore
from pytriton.proxy.inference_handler import InferenceHandler, _ResponsesIterator
from pytriton.proxy.types import Request
from tests.unit.utils import verify_equalness_of_dicts_with_ndarray

LOGGER = logging.getLogger("tests.unit.test_proxy_inference_handler")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


MODEL_CONFIG = TritonModelConfig(
    model_name="Foo",
    inputs=[
        TensorSpec(name="input1", dtype=np.int32, shape=(3,)),
        TensorSpec(name="input2", dtype=np.int32, shape=(3,)),
    ],
    outputs=[
        TensorSpec(name="output1", dtype=np.int32, shape=(3,)),
        TensorSpec(name="output2", dtype=np.int32, shape=(3,)),
    ],
)

DECOUPLED_MODEL_CONFIG = dataclasses.replace(MODEL_CONFIG, decoupled=True)


def _infer_fn(*_, **__):
    return [
        {"output1": np.array([1, 2, 3]), "output2": np.array([1, 2, 3])},
        {"output1": np.array([1, 2, 3]), "output2": np.array([1, 2, 3])},
    ]


def _infer_gen_fn(*_, **__):
    yield [
        {"output1": np.array([1]), "output2": np.array([1])},
        {"output1": np.array([1]), "output2": np.array([1])},
    ]
    yield [
        {"output1": np.array([2]), "output2": np.array([2])},
        {"output1": np.array([2]), "output2": np.array([2])},
    ]
    yield [
        {"output1": np.array([3]), "output2": np.array([3])},
        {"output1": np.array([3]), "output2": np.array([3])},
    ]


def _get_meta_requests_payload(_data_store_socket):
    tensor_store = TensorStore(_data_store_socket)
    try:
        LOGGER.debug(f"Connecting to tensor store {_data_store_socket} ...")
        tensor_store.connect()  # to already started tensor store
        requests = [
            Request({"input1": np.ones((128, 4), dtype="float32"), "input2": np.ones((128, 4), dtype="float32")}),
            Request({"input1": np.ones((128, 4), dtype="float32"), "input2": np.ones((128, 4), dtype="float32")}),
        ]
        input_arrays_with_coords = [
            (request_idx, input_name, tensor)
            for request_idx, request in enumerate(requests)
            for input_name, tensor in request.items()
        ]
        LOGGER.debug("Putting tensors to tensor store ...")
        tensor_ids = tensor_store.put([tensor for _, _, tensor in input_arrays_with_coords])
        requests_with_ids = [{}] * len(requests)
        for (request_idx, input_name, _), tensor_id in zip(input_arrays_with_coords, tensor_ids):
            requests_with_ids[request_idx][input_name] = tensor_id

        meta_requests = InferenceHandlerRequests(
            requests=[
                MetaRequestResponse(idx, data=request_with_ids, parameters=request.parameters)
                for idx, (request, request_with_ids) in enumerate(zip(requests, requests_with_ids))
            ]
        )
        LOGGER.debug(f"Return meta requests: {meta_requests}")
        return meta_requests.as_bytes()
    finally:
        tensor_store.close()


@pytest.mark.parametrize(
    "infer_fn,expected_response_lists,decoupled",
    [
        (_infer_fn, [_infer_fn()], False),
        (_infer_fn, [_infer_fn()], True),  # non-generator output should be also handled in decoupled mode
        (_infer_gen_fn, _infer_gen_fn(), True),
    ],
)
def test_responses_iterator(infer_fn, expected_response_lists, decoupled):
    responses = list(_ResponsesIterator(infer_fn(), decoupled=decoupled))
    for responses_list, expected_response_list in zip(responses, expected_response_lists):
        assert len(responses_list) == len(expected_response_list)
        for response, expected_response in zip(responses_list, expected_response_list):
            verify_equalness_of_dicts_with_ndarray(response, expected_response)


def test_responses_iterator_should_raise_error_when_generator_is_returned_for_nondecoupled_models():
    with pytest.raises(PyTritonRuntimeError, match="Results generator is not supported for non-decoupled models."):
        list(_ResponsesIterator(_infer_gen_fn(), decoupled=False))


def test_responses_iterator_could_iterate_only_once_on_non_generator_data():
    # it is usable to ensure that results are not consumed twice

    iterator = _ResponsesIterator(_infer_fn())
    responses1 = list(iterator)
    responses2 = list(iterator)

    assert len(responses1) == 1
    assert len(responses2) == 0


@pytest.mark.parametrize(
    "triton_model_config,infer_fn",
    [
        (MODEL_CONFIG, _infer_fn),
        (DECOUPLED_MODEL_CONFIG, _infer_gen_fn),
    ],
)
def test_proxy_throws_exception_when_validate_outputs_raise_an_error(tmp_path, mocker, triton_model_config, infer_fn):
    zmq_context = None
    inference_handler = None

    # simulate tensor store started by proxy backend
    data_store_socket = (tmp_path / "data_store.socket").as_posix()
    tensor_store = TensorStore(data_store_socket)  # authkey will be taken from current process
    try:
        tensor_store.start()  # start tensor store side process - this way InferenceHandler will create client for it
        mocker.patch(
            "pytriton.proxy.inference_handler.validate_outputs", side_effect=ValueError("Validate outputs error.")
        )
        zmq_context = zmq.Context()
        inference_handler = InferenceHandler(
            infer_fn,
            triton_model_config,
            shared_memory_socket=f"ipc://{tmp_path}/my",
            data_store_socket=data_store_socket,
            zmq_context=zmq_context,
            strict=False,
        )

        mock_recv = mocker.patch.object(inference_handler.zmq_context._socket_class, "recv")
        mock_recv.side_effect = functools.partial(_get_meta_requests_payload, data_store_socket)

        mocker.patch.object(inference_handler.zmq_context._socket_class, "send")  # do not send anything
        spy_send = mocker.spy(inference_handler.zmq_context._socket_class, "send")

        inference_handler.start()

        timeout_s = 1.0
        start_s = time.time()
        while not spy_send.called and time.time() - start_s < timeout_s:
            time.sleep(0.1)

        spy_send.assert_called()
        last_call = spy_send.mock_calls[-1]
        response_payload = last_call.args[0]
        response = json.loads(response_payload)
        error = response.get("error")
        assert error is not None and "Validate outputs error." in error
    finally:
        if inference_handler:
            inference_handler.stop()
            if inference_handler.is_alive():
                inference_handler.join(timeout=1)
        tensor_store.close()
        if zmq_context:
            zmq_context.term()
