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
import json
import logging
import time

import numpy as np
import zmq

from pytriton.model_config.triton_model_config import TensorSpec, TritonModelConfig
from pytriton.proxy.communication import InferenceHandlerRequests, MetaRequestResponse, TensorStore
from pytriton.proxy.inference_handler import InferenceHandler
from pytriton.proxy.types import Request

LOGGER = logging.getLogger("tests.unit.test_proxy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


MY_MODEL_CONFIG = TritonModelConfig(
    model_name="Foo",
    inputs=[
        TensorSpec(name="input1", dtype=np.float32, shape=(-1,)),
        TensorSpec(name="input2", dtype=np.float32, shape=(-1,)),
    ],
    outputs=[
        TensorSpec(name="output1", dtype=np.float32, shape=(-1,)),
        TensorSpec(name="output2", dtype=np.float32, shape=(-1,)),
    ],
)


def _patch_inference_handler__recv_send(mocker, inference_handler: InferenceHandler, json_payload: bytes):
    mock_recv = mocker.patch.object(inference_handler.zmq_context._socket_class, "recv")
    mock_recv.return_value = json_payload
    mocker.patch.object(inference_handler.zmq_context._socket_class, "send")
    spy_send = mocker.spy(inference_handler.zmq_context._socket_class, "send")
    return spy_send


def test_proxy_throws_exception_when_infer_func_returns_non_supported_type(tmp_path, mocker):
    zmq_context = None
    proxy = None

    input1 = np.ones((128, 4), dtype="float32")
    input2 = np.zeros((128, 4), dtype="float32")

    # simulate tensor store started by proxy backend
    data_store_socket = (tmp_path / "data_store.socket").as_posix()
    tensor_store = TensorStore(data_store_socket)  # authkey will be taken from current process

    try:

        def _infer_fn(_):
            return {"output1": np.array([1, 2, 3]), "output2": np.array([1, 2, 3])}

        tensor_store.start()  # start tensor store side process - this way InferenceHandler will create client for it
        requests = [Request({"input1": input1, "input2": input2})]
        input_arrays_with_coords = [
            (request_idx, input_name, tensor)
            for request_idx, request in enumerate(requests)
            for input_name, tensor in request.items()
        ]
        tensor_ids = tensor_store.put([tensor for _, _, tensor in input_arrays_with_coords])
        requests_with_ids = [{}] * len(requests)
        for (request_idx, input_name, _), tensor_id in zip(input_arrays_with_coords, tensor_ids):
            requests_with_ids[request_idx][input_name] = tensor_id

        meta_requests = InferenceHandlerRequests(
            requests=[
                MetaRequestResponse(data=request_with_ids, parameters=request.parameters)
                for request, request_with_ids in zip(requests, requests_with_ids)
            ]
        )

        zmq_context = zmq.Context()
        proxy = InferenceHandler(
            _infer_fn,
            MY_MODEL_CONFIG,
            shared_memory_socket=f"ipc://{tmp_path}/my",
            data_store_socket=data_store_socket,
            zmq_context=zmq_context,
        )
        spy_send = _patch_inference_handler__recv_send(mocker, proxy, meta_requests.as_bytes())
        proxy.start()

        start_s = time.time()
        while not spy_send.called and time.time() - start_s < 1:
            time.sleep(0.1)

        spy_send.assert_called()
        last_call = spy_send.mock_calls[-1]
        response_payload = last_call.args[0]
        response = json.loads(response_payload)
        error = response.get("error")
        assert error is not None and "Outputs returned by model callable must be list of request dicts" in error
    finally:
        if proxy:
            proxy.stop()
            if proxy.is_alive():
                proxy.join(timeout=1)
        tensor_store.close()
        if zmq_context:
            zmq_context.term()


def test_proxy_throws_exception_when_infer_func_returns_non_supported_output_item_type(tmp_path, mocker):
    input1 = np.ones((128, 4), dtype="float32")
    input2 = np.zeros((128, 4), dtype="float32")

    # tensor store client
    data_store_socket = (tmp_path / "data_store.socket").as_posix()
    tensor_store = TensorStore(data_store_socket)  # authkey will be taken from current process

    zmq_context = None
    proxy = None
    try:

        def _infer_fn(_):
            return [{"output1": [1, 2, 3], "output2": np.array([1, 2, 3])}]

        tensor_store.start()  # start tensor store side process - this way InferenceHandler will create client for it

        requests = [Request(data={"input1": input1, "input2": input2}, parameters={})]
        input_arrays_with_coords = [
            (request_idx, input_name, tensor)
            for request_idx, request in enumerate(requests)
            for input_name, tensor in request.items()
        ]
        tensor_ids = tensor_store.put([tensor for _, _, tensor in input_arrays_with_coords])
        requests_with_ids = [{}] * len(requests)
        for (request_idx, input_name, _), tensor_id in zip(input_arrays_with_coords, tensor_ids):
            requests_with_ids[request_idx][input_name] = tensor_id

        meta_requests = InferenceHandlerRequests(
            requests=[
                MetaRequestResponse(data=request_with_ids, parameters=request.parameters)
                for request, request_with_ids in zip(requests, requests_with_ids)
            ]
        )

        zmq_context = zmq.Context()
        proxy = InferenceHandler(
            _infer_fn,
            MY_MODEL_CONFIG,
            shared_memory_socket=f"ipc://{tmp_path}/my",
            data_store_socket=data_store_socket,
            zmq_context=zmq_context,
        )
        spy_send = _patch_inference_handler__recv_send(mocker, proxy, meta_requests.as_bytes())
        proxy.start()

        start_s = time.time()
        while not spy_send.called and time.time() - start_s < 1:
            time.sleep(0.1)

        spy_send.assert_called()
        last_call = spy_send.mock_calls[-1]
        response_payload = last_call.args[0]
        response = json.loads(response_payload)
        error = response.get("error")
        assert error is not None and "Not all values returned by model callable are numpy arrays" in error
    finally:
        if proxy:
            proxy.stop()
            if proxy.is_alive():
                proxy.join(timeout=1)
        tensor_store.close()
        if zmq_context:
            zmq_context.term()
