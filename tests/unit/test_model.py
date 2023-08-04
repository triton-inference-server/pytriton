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
import pathlib
import tempfile

import numpy as np

from pytriton.decorators import TritonContext, batch
from pytriton.model_config.tensor import Tensor
from pytriton.model_config.triton_model_config import TensorSpec
from pytriton.models.manager import ModelManager
from pytriton.models.model import Model, ModelConfig
from pytriton.proxy.communication import TensorStore
from pytriton.proxy.types import Request
from pytriton.server.model_repository import TritonModelRepository
from pytriton.utils.workspace import Workspace


def test_get_model_config_return_model_config_when_minimal_required_data(tmp_path):
    def infer_func(inputs):
        return inputs

    triton_context = TritonContext()
    workspace = Workspace(tmp_path / "workspace")
    model = Model(
        model_name="simple",
        model_version=2,
        inference_fn=infer_func,
        inputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        outputs=[
            Tensor(dtype=np.int32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128, batching=True),
        workspace=workspace,
        triton_context=triton_context,
        strict=False,
    )

    model_config = model._get_triton_model_config()

    assert model_config.model_name == "simple"
    assert model_config.model_version == 2

    assert model_config.batching is True
    assert model_config.max_batch_size == 128

    assert model_config.inputs == [
        TensorSpec(name="INPUT_1", dtype=np.float32, shape=(-1,)),
        TensorSpec(name="INPUT_2", dtype=np.float32, shape=(-1,)),
    ]

    assert model_config.outputs == [
        TensorSpec(name="OUTPUT_1", dtype=np.int32, shape=(-1,)),
    ]

    ipc_socket_path = workspace.path / "ipc_proxy_backend_simple"
    assert model_config.backend_parameters == {
        "shared-memory-socket": f"ipc://{ipc_socket_path.as_posix()}",
    }


def test_get_model_config_return_model_config_when_custom_names():
    def infer_func(inputs):
        return inputs

    triton_context = TritonContext()
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = pathlib.Path(tempdir)
        workspace = Workspace(tempdir / "workspace")
        model = Model(
            model_name="simple",
            model_version=2,
            inference_fn=infer_func,
            inputs=[
                Tensor(name="variable1", dtype=object, shape=(2, 1)),
                Tensor(name="variable2", dtype=np.float32().dtype, shape=(2, 1)),
            ],
            outputs=[
                Tensor(name="factorials", dtype=np.int32().dtype, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=128, batching=True),
            workspace=workspace,
            triton_context=triton_context,
            strict=False,
        )

        model_config = model._get_triton_model_config()

        assert model_config.model_name == "simple"
        assert model_config.model_version == 2

        assert model_config.batching is True
        assert model_config.max_batch_size == 128

        assert model_config.inputs == [
            TensorSpec(name="variable1", dtype=object, shape=(2, 1)),
            TensorSpec(name="variable2", dtype=np.float32, shape=(2, 1)),
        ]

        assert model_config.outputs == [
            TensorSpec(name="factorials", dtype=np.int32, shape=(-1,)),
        ]


def test_generate_model_create_model_store():
    def infer_func(inputs):
        return inputs

    triton_context = TritonContext()
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = pathlib.Path(tempdir)
        workspace = Workspace(tempdir / "workspace")
        model = Model(
            model_name="simple",
            model_version=2,
            inference_fn=infer_func,
            inputs=[
                Tensor(name="variable1", dtype=object, shape=(2, 1)),
                Tensor(name="variable2", dtype=np.float32, shape=(2, 1)),
            ],
            outputs=[
                Tensor(name="factorials", dtype=np.int32, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=128, batching=True),
            workspace=workspace,
            triton_context=triton_context,
            strict=False,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            model_repository = pathlib.Path(tempdir) / "model_repository"
            model_repository.mkdir()

            model.generate_model(model_repository)

            assert (model_repository / "simple").is_dir()
            assert (model_repository / "simple" / "config.pbtxt").is_file()

            assert (model_repository / "simple" / "2").is_dir()
            assert (model_repository / "simple" / "2" / "model.py").is_file()


def test_generate_models_with_same_names_and_different_versions_create_model_store():
    def infer_func(inputs):
        return inputs

    triton_context = TritonContext()
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = pathlib.Path(tempdir)
        workspace = Workspace(tempdir / "workspace")
        model1 = Model(
            model_name="simple",
            model_version=1,
            inference_fn=infer_func,
            inputs=[
                Tensor(name="variable1", dtype=object, shape=(2, 1)),
                Tensor(name="variable2", dtype=np.float32, shape=(2, 1)),
            ],
            outputs=[
                Tensor(name="factorials", dtype=np.int32, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=128, batching=True),
            workspace=workspace,
            triton_context=triton_context,
            strict=False,
        )
        model2 = Model(
            model_name="simple",
            model_version=2,
            inference_fn=infer_func,
            inputs=[
                Tensor(name="variable1", dtype=object, shape=(2, 1)),
                Tensor(name="variable2", dtype=np.float32, shape=(2, 1)),
            ],
            outputs=[
                Tensor(name="factorials", dtype=np.int32, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=128, batching=True),
            workspace=workspace,
            triton_context=triton_context,
            strict=False,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            model_repository = pathlib.Path(tempdir) / "model_repository"
            model_repository.mkdir()

            model1.generate_model(model_repository)
            model2.generate_model(model_repository)

            assert (model_repository / "simple").is_dir()
            assert (model_repository / "simple" / "config.pbtxt").is_file()

            assert (model_repository / "simple" / "1").is_dir()
            assert (model_repository / "simple" / "1" / "model.py").is_file()

            assert (model_repository / "simple" / "2").is_dir()
            assert (model_repository / "simple" / "2" / "model.py").is_file()


def test_setup_create_proxy_backend_connection(tmp_path):
    def infer_func(inputs):
        return inputs

    triton_context = TritonContext()
    workspace = Workspace(tmp_path / "workspace")
    tensor_store = TensorStore(workspace.path / "data_store.sock")
    model = Model(
        model_name="simple",
        model_version=2,
        inference_fn=infer_func,
        inputs=[
            Tensor(name="variable1", dtype=object, shape=(2, 1)),
            Tensor(name="variable2", dtype=np.float32, shape=(2, 1)),
        ],
        outputs=[
            Tensor(name="factorials", dtype=np.int32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128, batching=True),
        workspace=workspace,
        triton_context=triton_context,
        strict=False,
    )

    try:
        tensor_store.start()
        model.setup()
        assert len(model._inference_handlers) == 1
    finally:
        model.clean()
        tensor_store.close()


def test_setup_can_be_called_multiple_times(tmp_path):
    def infer_func(inputs):
        return inputs

    triton_context = TritonContext()
    workspace = Workspace(tmp_path / "workspace")
    tensor_store = TensorStore(workspace.path / "data_store.sock")
    model = Model(
        model_name="simple",
        model_version=2,
        inference_fn=infer_func,
        inputs=[
            Tensor(name="variable1", dtype=object, shape=(2, 1)),
            Tensor(name="variable2", dtype=np.float32, shape=(2, 1)),
        ],
        outputs=[
            Tensor(name="factorials", dtype=np.int32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128, batching=True),
        workspace=workspace,
        triton_context=triton_context,
        strict=False,
    )

    try:
        tensor_store.start()
        model.setup()
        assert len(model._inference_handlers) == 1
        python_backend1 = model._inference_handlers[0]

        assert python_backend1 is not None

        model.setup()
        assert len(model._inference_handlers) == 1
        python_backend2 = model._inference_handlers[0]

        assert python_backend2 is not None
        assert python_backend1 == python_backend2

    finally:
        model.clean()
        tensor_store.close()


def test_clean_remove_proxy_backend_connection(tmp_path):
    def infer_func(inputs):
        return inputs

    triton_context = TritonContext()
    workspace = Workspace(tmp_path / "workspace")
    tensor_store = TensorStore(workspace.path / "data_store.sock")
    model = Model(
        model_name="simple",
        model_version=2,
        inference_fn=infer_func,
        inputs=[
            Tensor(name="variable1", dtype=object, shape=(2, 1)),
            Tensor(name="variable2", dtype=np.float32, shape=(2, 1)),
        ],
        outputs=[
            Tensor(name="factorials", dtype=np.int32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128, batching=True),
        workspace=workspace,
        triton_context=triton_context,
        strict=False,
    )

    try:
        tensor_store.start()
        model.setup()
    finally:
        model.clean()
        tensor_store.close()
    assert len(model._inference_handlers) == 0


def test_clean_can_be_called_multiple_times(tmp_path):
    def infer_func(inputs):
        return inputs

    triton_context = TritonContext()
    workspace = Workspace(tmp_path / "workspace")
    tensor_store = TensorStore(workspace.path / "data_store.sock")
    model = Model(
        model_name="simple",
        model_version=2,
        inference_fn=infer_func,
        inputs=[
            Tensor(name="variable1", dtype=object, shape=(2, 1)),
            Tensor(name="variable2", dtype=np.float32, shape=(2, 1)),
        ],
        outputs=[
            Tensor(name="factorials", dtype=np.int32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128, batching=True),
        workspace=workspace,
        triton_context=triton_context,
        strict=False,
    )

    try:
        tensor_store.start()
        model.setup()
        model.clean()
        model.clean()
        assert len(model._inference_handlers) == 0
    finally:
        tensor_store.close()


def test_is_alive_return_false_when_model_not_setup(tmp_path):
    def infer_func(inputs):
        return inputs

    triton_context = TritonContext()
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = pathlib.Path(tempdir)
        workspace = Workspace(tempdir / "workspace")
        model = Model(
            model_name="simple",
            model_version=2,
            inference_fn=infer_func,
            inputs=[
                Tensor(name="variable1", dtype=object, shape=(2, 1)),
                Tensor(name="variable2", dtype=np.float32, shape=(2, 1)),
            ],
            outputs=[
                Tensor(name="factorials", dtype=np.int32, shape=(-1,)),
            ],
            config=ModelConfig(max_batch_size=128, batching=True),
            workspace=workspace,
            triton_context=triton_context,
            strict=False,
        )

    assert not model.is_alive()


def test_is_alive_return_true_when_model_is_setup(tmp_path):
    def infer_func(inputs):
        return inputs

    triton_context = TritonContext()
    workspace = Workspace(tmp_path / "workspace")
    tensor_store = TensorStore(workspace.path / "data_store.sock")
    model = Model(
        model_name="simple",
        model_version=2,
        inference_fn=infer_func,
        inputs=[
            Tensor(name="variable1", dtype=object, shape=(2, 1)),
            Tensor(name="variable2", dtype=np.float32, shape=(2, 1)),
        ],
        outputs=[
            Tensor(name="factorials", dtype=np.int32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128, batching=True),
        workspace=workspace,
        triton_context=triton_context,
        strict=False,
    )

    try:
        tensor_store.start()
        model.setup()
        assert model.is_alive()
        assert len(model._inference_handlers) == 1
    finally:
        model.clean()
        tensor_store.close()


def test_triton_context_injection(tmp_path):
    class Multimodel:
        @batch
        def infer1(self, variable1):
            return [variable1]

        @batch
        def infer2(self, variable2):
            return [variable2]

    m = Multimodel()

    @batch
    def infer_func(variable3):
        return [variable3]

    triton_context = TritonContext()
    workspace = Workspace(tmp_path / "workspace")
    tensor_store = TensorStore(workspace.path / "data_store.sock")
    model1 = Model(
        model_name="simple1",
        model_version=1,
        inference_fn=m.infer1,
        inputs=[
            Tensor(name="variable1", dtype=np.int32, shape=(2, 1)),
        ],
        outputs=[
            Tensor(name="out1", dtype=np.int32, shape=(2, 1)),
        ],
        config=ModelConfig(max_batch_size=128, batching=True),
        workspace=workspace,
        triton_context=triton_context,
        strict=False,
    )
    model2 = Model(
        model_name="simple2",
        model_version=1,
        inference_fn=m.infer2,
        inputs=[
            Tensor(name="variable2", dtype=np.int32, shape=(2, 1)),
        ],
        outputs=[
            Tensor(name="out2", dtype=np.int32, shape=(2, 1)),
        ],
        config=ModelConfig(max_batch_size=128, batching=True),
        workspace=workspace,
        triton_context=triton_context,
        strict=False,
    )
    model3 = Model(
        model_name="simple3",
        model_version=1,
        inference_fn=infer_func,
        inputs=[
            Tensor(name="variable3", dtype=np.int32, shape=(2, 1)),
        ],
        outputs=[
            Tensor(name="out3", dtype=np.int32, shape=(2, 1)),
        ],
        config=ModelConfig(max_batch_size=128, batching=True),
        workspace=workspace,
        triton_context=triton_context,
        strict=False,
    )

    tr = TritonModelRepository(path=None, workspace=workspace)
    manager = ModelManager(tr)
    try:
        tensor_store.start()
        manager.add_model(model1)
        manager.add_model(model2)
        manager.add_model(model3)
        manager.create_models()

        input_requests1 = [Request({"variable1": np.array([[7, 5], [8, 6]])}, {})]
        input_requests2 = [Request({"variable2": np.array([[1, 2], [1, 2], [11, 12]])}, {})]
        input_requests3 = [Request({"variable3": np.array([[1, 2]])}, {})]

        def assert_inputs_properly_mapped_to_outputs(expected_out_name, outputs, input_request_arr):
            assert len(outputs) == 1
            assert expected_out_name in outputs[0]
            assert outputs[0][expected_out_name].shape == input_request_arr.shape
            assert np.array_equal(outputs[0][expected_out_name], input_request_arr)

        outputs1 = m.infer1(input_requests1)
        assert_inputs_properly_mapped_to_outputs("out1", outputs1, input_requests1[0]["variable1"])

        outputs2 = m.infer2(input_requests2)
        assert_inputs_properly_mapped_to_outputs("out2", outputs2, input_requests2[0]["variable2"])

        outputs3 = infer_func(input_requests3)
        assert_inputs_properly_mapped_to_outputs("out3", outputs3, input_requests3[0]["variable3"])

        outputs1 = m.infer1(input_requests1)
        assert_inputs_properly_mapped_to_outputs("out1", outputs1, input_requests1[0]["variable1"])

        outputs3 = infer_func(input_requests3)
        assert_inputs_properly_mapped_to_outputs("out3", outputs3, input_requests3[0]["variable3"])
    finally:
        manager.clean()
        tensor_store.close()
