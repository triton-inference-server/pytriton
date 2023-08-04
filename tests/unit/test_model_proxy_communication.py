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
import multiprocessing
import pathlib
import sys
import time
import traceback
from unittest.mock import Mock

import numpy as np
import pytest
import zmq

from pytriton.model_config.generator import ModelConfigGenerator
from pytriton.model_config.triton_model_config import TensorSpec, TritonModelConfig
from pytriton.proxy.communication import TensorStore
from pytriton.proxy.types import Request
from pytriton.triton import TRITONSERVER_DIST_DIR

LOGGER = logging.getLogger("tests.test_model_error_handling")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
    level=logging.INFO,
)


def test_model_throws_exception(tmp_path, mocker):

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
        pb_utils.Logger = Mock()

        from pytriton.proxy.inference_handler import InferenceHandler
        from pytriton.proxy.model import TritonPythonModel
        from pytriton.utils.workspace import Workspace

        def infer_fn(_):
            # Wrapper raises division by zero error
            time.sleep(0.2)
            return 2 / 0

        model_name = "model1"
        workspace = Workspace(pathlib.Path(tmp_path) / "w")
        ipc_socket_path = workspace.path / f"proxy_{model_name}.ipc"
        shared_memory_socket = f"ipc://{ipc_socket_path.as_posix()}"
        data_store_socket = (workspace.path / "data_store.socket").as_posix()

        model_config = TritonModelConfig(
            model_name=model_name,
            inputs=[TensorSpec(name="input1", dtype=np.float32, shape=(-1,))],
            outputs=[TensorSpec(name="output1", dtype=np.float32, shape=(-1,))],
            backend_parameters={"shared-memory-socket": shared_memory_socket},
        )
        model_config_json_payload = json.dumps(ModelConfigGenerator(model_config).get_config()).encode("utf-8")

        zmq_context = zmq.Context()

        authkey = multiprocessing.current_process().authkey
        tensor_store = TensorStore(data_store_socket, authkey)
        tensor_store.start()

        authkey = base64.b64encode(authkey).decode("utf-8")
        instance_data = {
            "shared-memory-socket": shared_memory_socket,
            "data-store-socket": data_store_socket,
            "auth-key": authkey,
        }

        with mocker.patch.object(TritonPythonModel, "_get_instance_data", return_value=instance_data):
            backend_initialization_args = {"model_config": model_config_json_payload}
            backend_model = TritonPythonModel()
            backend_model.initialize(backend_initialization_args)

            inference_handler = InferenceHandler(
                infer_fn,
                model_config,
                shared_memory_socket=shared_memory_socket,
                data_store_socket=data_store_socket,
                zmq_context=zmq_context,
                strict=False,
            )
            inference_handler.start()

            inputs = [Request({"input1": np.array([[1, 2, 3]], dtype=np.float32)}, {})]

            try:
                result = backend_model._exec_requests(inputs)
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
                zmq_context.term()
                inference_handler.stop()
                backend_model.finalize()

    finally:
        sys.path.pop()
        if "pb_utils" in locals() and hasattr(pb_utils, "TritonModelException"):  # pytype: disable=name-error
            delattr(pb_utils, "TritonModelException")  # pytype: disable=name-error

        print("sys.path cleaned-up")  # noqa: T201
        for entry in sys.path:
            print(f"  {entry}")  # noqa: T201
