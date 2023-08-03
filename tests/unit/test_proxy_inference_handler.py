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
from pytriton.proxy.communication import InferenceHandlerRequest, MetaRequestResponse, ShmManager
from pytriton.proxy.inference_handler import InferenceHandler
from pytriton.proxy.types import Request

LOGGER = logging.getLogger("tests.unit.test_proxy_inference_handler")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


MY_MODEL_CONFIG = TritonModelConfig(
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


def _patch_inference_handler__recv_send(mocker, inference_handler: InferenceHandler, json_payload: bytes):
    mock_recv = mocker.patch.object(inference_handler.zmq_context._socket_class, "recv")
    mock_recv.return_value = json_payload
    mocker.patch.object(inference_handler.zmq_context._socket_class, "send")
    spy_send = mocker.spy(inference_handler.zmq_context._socket_class, "send")
    return spy_send


def test_proxy_throws_exception_when_validate_outputs_raise_an_error(tmp_path, mocker):
    zmq_context = None
    proxy = None

    input1 = np.ones((128, 4), dtype="float32")
    input2 = np.zeros((128, 4), dtype="float32")

    shm_manager = ShmManager()

    inputs = [Request({"input1": input1, "input2": input2}, {})]
    inputs_infos = shm_manager.to_shm(inputs, lambda data, req: MetaRequestResponse(data, req.parameters))

    try:

        def _infer_fn(_):
            return {"output1": np.array([1, 2, 3]), "output2": np.array([1, 2, 3])}

        with mocker.patch(
            "pytriton.proxy.inference_handler.validate_outputs", side_effect=ValueError("Validate outputs error.")
        ):
            zmq_context = zmq.Context()
            proxy = InferenceHandler(_infer_fn, MY_MODEL_CONFIG, f"ipc://{tmp_path}/my", zmq_context, strict=False)
            spy_send = _patch_inference_handler__recv_send(
                mocker,
                proxy,
                InferenceHandlerRequest(requests=inputs_infos, memory_name=shm_manager.memory_name()).as_bytes(),
            )
            proxy.start()

            start_s = time.time()
            while not spy_send.called and time.time() - start_s < 1:
                time.sleep(0.1)

            spy_send.assert_called()
            last_call = spy_send.mock_calls[-1]
            response_payload = last_call.args[0]
            response = json.loads(response_payload)
            error = response.get("error")
            assert error is not None and "Validate outputs error." in error
    finally:
        if proxy:
            proxy.stop()
            if proxy.is_alive():
                proxy.join(timeout=1)
        shm_manager.dispose()
        if zmq_context:
            zmq_context.term()
