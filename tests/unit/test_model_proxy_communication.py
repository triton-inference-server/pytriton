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

import base64
import json
import logging
import pathlib
import sys
import time
import traceback

import numpy as np
import pytest

from pytriton.model_config.generator import ModelConfigGenerator
from pytriton.model_config.triton_model_config import TensorSpec, TritonModelConfig
from pytriton.proxy.communication import get_config_from_handshake_server
from pytriton.proxy.data import TensorStoreSerializerDeserializer
from pytriton.proxy.inference import RequestsResponsesConnector
from pytriton.proxy.validators import TritonResultsValidator
from pytriton.triton import TRITONSERVER_DIST_DIR

LOGGER = logging.getLogger("tests.test_model_error_handling")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
    level=logging.INFO,
)


class Tensor:
    def __init__(self, name, data):
        self._name = name
        self._data = data

    def name(self):
        return self._name

    def as_numpy(self):
        return self._data


class InferenceRequest:
    def __init__(self, model_name, inputs, requested_output_names, parameters=None):
        self.model_name = model_name
        self._inputs = inputs
        self._requested_output_names = requested_output_names
        self._parameters = parameters or {}

    def inputs(self):
        return self._inputs

    def parameters(self):
        return json.dumps(self._parameters)

    def get_response_sender(self):
        return None

    def requested_output_names(self):
        return self._requested_output_names


def _error_infer_fn(*_, **__):
    # Wrapper raises division by zero error
    time.sleep(0.2)
    return 2 / 0


def _error_infer_gen_fn(*_, **__):
    # Wrapper raises division by zero error
    time.sleep(0.2)
    raise RuntimeError("division by zero")


def _get_proxy_backend(model_config):
    from pytriton.proxy.model import TritonPythonModel

    model_config_json_payload = json.dumps(ModelConfigGenerator(model_config).get_config()).encode("utf-8")
    backend_initialization_args = {
        "model_config": model_config_json_payload,
        "model_name": model_config.model_name,
        "model_instance_name": f"{model_config.model_name}_0_0",
    }

    backend_model = None
    try:
        backend_model = TritonPythonModel()
        backend_model.initialize(backend_initialization_args)
        return backend_model
    except Exception:
        if backend_model:
            backend_model.finalize()
        raise


@pytest.mark.parametrize(
    "infer_fn,decoupled",
    [
        (_error_infer_fn, False),
    ],
)
def test_model_throws_exception(tmp_path, mocker, infer_fn, decoupled):
    # add python backend folder to find triton_python_backend_utils from model.py
    python_backend_path = TRITONSERVER_DIST_DIR / "backends" / "python"
    sys.path.append(str(python_backend_path))

    print("sys.path updated")  # noqa: T201
    for entry in sys.path:
        print(f"  {entry}")  # noqa: T201

    try:
        import triton_python_backend_utils as pb_utils  # pytype: disable=import-error

        # add TritonModelException to pb_utils for test (python backend does this in C++ code)
        pb_utils.TritonModelException = RuntimeError
        pb_utils.Logger = mocker.Mock()

        from pytriton.proxy.inference import InferenceHandler
        from pytriton.utils.workspace import Workspace

        model_name = "model1"
        workspace = Workspace(pathlib.Path(tmp_path) / "w")

        model_config = TritonModelConfig(
            model_name=model_name,
            inputs=[TensorSpec(name="input1", dtype=np.float32, shape=(-1,))],
            outputs=[TensorSpec(name="output1", dtype=np.float32, shape=(-1,))],
            backend_parameters={"workspace-path": workspace.path.as_posix()},
            decoupled=decoupled,
        )

        backend_model = _get_proxy_backend(model_config)

        validator = TritonResultsValidator(model_config, True)
        inference_handler_config_path = workspace.path / f"{model_name}-config.sock"
        inference_handler_config = get_config_from_handshake_server(inference_handler_config_path)

        serializer_deserializer = TensorStoreSerializerDeserializer()
        serializer_deserializer.connect(
            inference_handler_config["data_socket"],
            base64.decodebytes(inference_handler_config["authkey"].encode("ascii")),
        )
        request_server_socket = workspace.path / f"{model_name}_0_0-server.sock"
        request_server_socket = f"ipc://{request_server_socket.as_posix()}"
        requests_respones_connector = RequestsResponsesConnector(
            url=request_server_socket,
            serializer_deserializer=serializer_deserializer,
        )
        requests_respones_connector.start()
        inference_handler = InferenceHandler(
            infer_fn,
            requests_responses_connector=requests_respones_connector,
            validator=validator,
            name=f"inference_handler-{model_name}",
        )
        inference_handler.start()

        requests = [
            InferenceRequest(
                model_name=model_name,
                inputs=[Tensor("input1", np.array([[1, 2, 3]], dtype=np.float32))],
                requested_output_names=["output1"],
            ),
        ]

        try:
            result = backend_model.execute(requests)
            pytest.fail(f"Model raised exception, but exec_batch passed - result: {result}")
        except pb_utils.TritonModelException:  # pytype: disable=module-attr
            LOGGER.info("Inference exception")
            msg = traceback.format_exc()
            LOGGER.info(msg)
            assert "division by zero" in msg
        except Exception:
            msg = traceback.format_exc()
            pytest.fail(f"Wrong exception raised: {msg}")
        finally:
            inference_handler.stop()
            requests_respones_connector.close()
            backend_model.finalize()

    finally:
        sys.path.pop()
        if "pb_utils" in locals() and hasattr(pb_utils, "TritonModelException"):  # pytype: disable=name-error
            delattr(pb_utils, "TritonModelException")  # pytype: disable=name-error

        print("sys.path cleaned-up")  # noqa: T201
        for entry in sys.path:
            print(f"  {entry}")  # noqa: T201
