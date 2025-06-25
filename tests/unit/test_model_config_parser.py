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

import numpy as np
import pytest

from pytriton.exceptions import PyTritonModelConfigError
from pytriton.model_config.common import DeviceKind, DynamicBatcher, TimeoutAction
from pytriton.model_config.parser import ModelConfigParser
from pytriton.model_config.triton_model_config import ResponseCache, TensorSpec

from .common import full_model_config

common_model_config = {
    "backend": "python",
    "instance_group": [{"kind": "KIND_CPU"}],
    "parameters": {"workspace-path": {"string_value": "/tmp"}},
}

invalid_model_config = {
    **common_model_config,
    **{
        "name": "minimal",
        "input": [
            {"data_type": "TYPE_FLOAT32", "dims": ["-1"], "name": "INPUT_0"},
        ],
        "output": [
            {"data_type": "INT32", "dims": ["-1"], "name": "OUTPUT_0"},
        ],
    },
}

minimal_model_config = {
    **common_model_config,
    **{
        "name": "minimal",
        "input": [
            {"data_type": "TYPE_INT32", "dims": ["-1"], "name": "INPUT_0"},
        ],
        "output": [
            {"data_type": "TYPE_INT32", "dims": ["-1"], "name": "OUTPUT_0"},
        ],
    },
}

minimal_response_model_config = {
    **common_model_config,
    **{
        "name": "minimal",
        "input": [
            {"data_type": "TYPE_INT32", "dims": ["-1"], "name": "INPUT_0"},
        ],
        "output": [
            {"data_type": "TYPE_INT32", "dims": ["-1"], "name": "OUTPUT_0"},
        ],
        "response_cache": {"enable": True},
    },
}

simple_add_model_config = {
    **common_model_config,
    **{
        "name": "add",
        "max_batch_size": 16,
        "input": [
            {"data_type": "TYPE_INT32", "dims": ["-1"], "name": "INPUT_0"},
            {"data_type": "TYPE_FP32", "dims": ["-1"], "name": "INPUT_1"},
        ],
        "output": [
            {"data_type": "TYPE_INT32", "dims": ["-1"], "name": "OUTPUT_0"},
        ],
        "dynamic_batching": {},
    },
}

string_model_config = {
    **common_model_config,
    **{
        "name": "string",
        "max_batch_size": 16,
        "input": [
            {"data_type": "TYPE_STRING", "dims": ["-1"], "name": "INPUT_0"},
        ],
        "output": [
            {"data_type": "TYPE_STRING", "dims": ["-1"], "name": "OUTPUT_0"},
        ],
    },
}

add_model_config_with_model_not_supporting_batching = {
    **common_model_config,
    # no max_batch_size and dynamic_batching keys
    **{
        "name": "add",
        "input": [
            {"data_type": "TYPE_INT32", "dims": ["1"], "name": "INPUT_0"},
            {"data_type": "TYPE_INT32", "dims": ["1"], "name": "INPUT_1"},
        ],
        "output": [
            {"data_type": "TYPE_INT32", "dims": ["1"], "name": "OUTPUT_0"},
        ],
    },
}

add_model_config_without_dynamic_batching = {
    **common_model_config,
    "max_batch_size": 16,
    **{
        "name": "add",
        "input": [
            {"data_type": "TYPE_INT32", "dims": ["1"], "name": "INPUT_0"},
            {"data_type": "TYPE_INT32", "dims": ["1"], "name": "INPUT_1"},
        ],
        "output": [
            {"data_type": "TYPE_INT32", "dims": ["1"], "name": "OUTPUT_0"},
        ],
    },
}

add_model_config_with_simple_dynamic_batching = {
    **common_model_config,
    **{
        "name": "add",
        "max_batch_size": 16,
        "input": [
            {"data_type": "TYPE_INT32", "dims": ["-1"], "name": "INPUT_0"},
            {"data_type": "TYPE_INT32", "dims": ["-1"], "name": "INPUT_1"},
        ],
        "output": [
            {"data_type": "TYPE_INT32", "dims": ["-1"], "name": "OUTPUT_0"},
        ],
        "dynamic_batching": {
            "max_queue_delay_microseconds": 100,
            "preferred_batch_size": [64, 128],
        },
    },
}

add_model_config_with_advanced_dynamic_batching = {
    **common_model_config,
    **{
        "name": "add",
        "max_batch_size": 16,
        "input": [
            {"data_type": "TYPE_INT32", "dims": ["-1"], "name": "INPUT_0"},
            {"data_type": "TYPE_INT32", "dims": ["-1"], "name": "INPUT_1"},
        ],
        "output": [
            {"data_type": "TYPE_INT32", "dims": ["-1"], "name": "OUTPUT_0"},
        ],
        "dynamic_batching": {
            "max_queue_delay_microseconds": 100,
            "preferred_batch_size": [64, 128],
            "preserve_ordering": True,
            "response_cache": True,
            "priority_levels": 2,
            "default_priority_level": 1,
            "default_queue_policy": {
                "timeout_action": "DELAY",
                "default_timeout_microseconds": 100,
                "allow_timeout_override": True,
                "max_queue_size": 10,
            },
            "priority_queue_policy": {
                1: {
                    "timeout_action": "DELAY",
                    "default_timeout_microseconds": 100,
                    "allow_timeout_override": True,
                    "max_queue_size": 10,
                },
                2: {
                    "timeout_action": "REJECT",
                    "default_timeout_microseconds": 1000,
                    "allow_timeout_override": False,
                    "max_queue_size": 2,
                },
            },
        },
    },
}


def test_rewrite_io_spec_raise_error_when_empty_dict():
    with pytest.raises(PyTritonModelConfigError, match="Name for input at index 0 not provided."):
        ModelConfigParser.rewrite_io_spec({}, io_type="input", idx=0)


def test_rewrite_io_spec_raise_error_when_no_data_type():
    with pytest.raises(PyTritonModelConfigError, match="Data type for input with name `input` not defined."):
        ModelConfigParser.rewrite_io_spec({"name": "input"}, io_type="input", idx=0)


def test_rewrite_io_spec_raise_error_when_no_invalid_data_type():
    with pytest.raises(
        PyTritonModelConfigError,
        match="Invalid data type `FLOAT32` for input with name `input` not defined. The expected name is TYPE_{type}.",
    ):
        ModelConfigParser.rewrite_io_spec(
            {
                "name": "input",
                "data_type": "FLOAT32",
            },
            io_type="input",
            idx=0,
        )


def test_rewrite_io_spec_raise_error_when_unsupported_data_type():
    with pytest.raises(
        PyTritonModelConfigError,
        match="Invalid data type `FLOAT32` for input with name `input` not defined. The expected name is TYPE_{type}.",
    ):
        ModelConfigParser.rewrite_io_spec(
            {
                "name": "input",
                "data_type": "FLOAT32",
            },
            io_type="input",
            idx=0,
        )


def test_rewrite_io_spec_raise_error_when_no_dimension():
    with pytest.raises(PyTritonModelConfigError, match="Dimension for input with name `input` not defined."):
        ModelConfigParser.rewrite_io_spec(
            {"name": "input", "data_type": "TYPE_FP32", "dims": None},
            io_type="input",
            idx=0,
        )


def test_rewrite_io_spec_return_tensor_spec_when_valid_data():
    tensor_spec = ModelConfigParser.rewrite_io_spec(
        {"name": "input", "data_type": "TYPE_FP32", "dims": [1]},
        io_type="input",
        idx=0,
    )

    assert tensor_spec == TensorSpec(name="input", dtype=np.float32, shape=(1,))


def test_parse_from_dict_raise_error_when_invalid_data_type_in_config():
    with pytest.raises(
        PyTritonModelConfigError, match="Unsupported data type `TYPE_FLOAT32` for input with name `INPUT_0`"
    ):
        ModelConfigParser.from_dict(model_config_dict=invalid_model_config)


def test_parse_from_dict_return_model_config_when_minimal_config_used():
    model_config = ModelConfigParser.from_dict(model_config_dict=minimal_model_config)
    assert model_config.model_name == "minimal"
    assert model_config.max_batch_size == 0
    assert model_config.batching is False

    assert len(model_config.instance_group) == 1

    device_kind = list(model_config.instance_group.keys())[0]
    device_count = list(model_config.instance_group.values())[0]
    assert device_kind.value == DeviceKind.KIND_CPU.value
    assert device_count is None

    assert not model_config.decoupled

    assert model_config.backend_parameters == {"workspace-path": "/tmp"}
    assert model_config.inputs[0] == TensorSpec(name="INPUT_0", dtype=np.int32, shape=(-1,))
    assert model_config.outputs[0] == TensorSpec(name="OUTPUT_0", dtype=np.int32, shape=(-1,))

    assert model_config.batcher is None


def test_parse_from_dict_return_model_config_when_simple_config_used():
    model_config = ModelConfigParser.from_dict(model_config_dict=simple_add_model_config)
    assert model_config.model_name == "add"
    assert model_config.max_batch_size == 16
    assert model_config.batching is True

    assert len(model_config.instance_group) == 1

    device_kind = list(model_config.instance_group.keys())[0]
    device_count = list(model_config.instance_group.values())[0]
    assert device_kind.value == DeviceKind.KIND_CPU.value
    assert device_count is None

    assert not model_config.decoupled

    assert model_config.backend_parameters == {"workspace-path": "/tmp"}
    assert model_config.inputs == [
        TensorSpec(name="INPUT_0", dtype=np.int32, shape=(-1,)),
        TensorSpec(name="INPUT_1", dtype=np.float32, shape=(-1,)),
    ]
    assert model_config.outputs == [TensorSpec(name="OUTPUT_0", dtype=np.int32, shape=(-1,))]

    assert model_config.batcher == DynamicBatcher()


def test_parse_from_dict_return_model_config_when_string_config_used():
    model_config = ModelConfigParser.from_dict(model_config_dict=string_model_config)
    assert model_config.model_name == "string"
    assert model_config.max_batch_size == 16
    assert model_config.batching is True

    assert len(model_config.instance_group) == 1

    device_kind = list(model_config.instance_group.keys())[0]
    device_count = list(model_config.instance_group.values())[0]
    assert device_kind.value == DeviceKind.KIND_CPU.value
    assert device_count is None

    assert not model_config.decoupled

    assert model_config.backend_parameters == {"workspace-path": "/tmp"}
    assert model_config.inputs == [
        TensorSpec(name="INPUT_0", dtype=np.bytes_, shape=(-1,)),
    ]
    assert model_config.outputs == [TensorSpec(name="OUTPUT_0", dtype=np.bytes_, shape=(-1,))]

    assert model_config.batcher is None


def test_parse_from_dict_return_model_config_when_add_model_without_batching_used():
    model_config = ModelConfigParser.from_dict(model_config_dict=add_model_config_with_model_not_supporting_batching)
    assert model_config.model_name == "add"
    assert model_config.max_batch_size == 0
    assert model_config.batching is False

    assert len(model_config.instance_group) == 1

    device_kind = list(model_config.instance_group.keys())[0]
    device_count = list(model_config.instance_group.values())[0]
    assert device_kind.value == DeviceKind.KIND_CPU.value
    assert device_count is None

    assert not model_config.decoupled

    assert model_config.backend_parameters == {"workspace-path": "/tmp"}
    assert model_config.inputs == [
        TensorSpec(name="INPUT_0", dtype=np.int32, shape=(1,)),
        TensorSpec(name="INPUT_1", dtype=np.int32, shape=(1,)),
    ]
    assert model_config.outputs == [TensorSpec(name="OUTPUT_0", dtype=np.int32, shape=(1,))]

    assert model_config.batcher is None


def test_parse_from_dict_return_model_config_when_decoupled_execution_is_enabled():
    model_config_dict = {
        **minimal_model_config,
        **{
            "model_transaction_policy": {"decoupled": True},
        },
    }

    model_config = ModelConfigParser.from_dict(model_config_dict=model_config_dict)
    assert model_config.decoupled


def test_parse_from_dict_return_model_config_when_decoupled_execution_is_explicitly_disabled():
    model_config_dict = {
        **minimal_model_config,
        **{
            "model_transaction_policy": {"decoupled": False},
        },
    }

    model_config = ModelConfigParser.from_dict(model_config_dict=model_config_dict)
    assert not model_config.decoupled


def test_parse_from_dict_return_model_config_when_add_model_without_dynamic_batching_used():
    model_config = ModelConfigParser.from_dict(model_config_dict=add_model_config_without_dynamic_batching)
    assert model_config.model_name == "add"
    assert model_config.max_batch_size == 16
    assert model_config.batching is True

    assert len(model_config.instance_group) == 1

    device_kind = list(model_config.instance_group.keys())[0]
    device_count = list(model_config.instance_group.values())[0]
    assert device_kind.value == DeviceKind.KIND_CPU.value
    assert device_count is None

    assert not model_config.decoupled

    assert model_config.backend_parameters == {"workspace-path": "/tmp"}
    assert model_config.inputs == [
        TensorSpec(name="INPUT_0", dtype=np.int32, shape=(1,)),
        TensorSpec(name="INPUT_1", dtype=np.int32, shape=(1,)),
    ]
    assert model_config.outputs == [TensorSpec(name="OUTPUT_0", dtype=np.int32, shape=(1,))]

    assert model_config.batcher is None


def test_parse_from_dict_return_model_config_when_add_model_with_simple_dynamic_batching_used():
    model_config = ModelConfigParser.from_dict(model_config_dict=add_model_config_with_simple_dynamic_batching)
    assert model_config.model_name == "add"
    assert model_config.max_batch_size == 16
    assert model_config.batching is True

    assert len(model_config.instance_group) == 1

    device_kind = list(model_config.instance_group.keys())[0]
    device_count = list(model_config.instance_group.values())[0]
    assert device_kind.value == DeviceKind.KIND_CPU.value
    assert device_count is None

    assert not model_config.decoupled

    assert model_config.backend_parameters == {"workspace-path": "/tmp"}
    assert model_config.inputs == [
        TensorSpec(name="INPUT_0", dtype=np.int32, shape=(-1,)),
        TensorSpec(name="INPUT_1", dtype=np.int32, shape=(-1,)),
    ]
    assert model_config.outputs == [
        TensorSpec(name="OUTPUT_0", dtype=np.int32, shape=(-1,)),
    ]
    assert model_config.batcher.preferred_batch_size == [64, 128]
    assert model_config.batcher.max_queue_delay_microseconds == 100
    assert model_config.batcher.preserve_ordering is False

    assert model_config.batcher.priority_levels == 0
    assert model_config.batcher.default_priority_level == 0
    assert model_config.batcher.default_queue_policy is None
    assert model_config.batcher.priority_queue_policy is None


def test_parse_from_dict_return_model_config_when_add_model_with_advanced_batcher_used():
    model_config = ModelConfigParser.from_dict(model_config_dict=add_model_config_with_advanced_dynamic_batching)
    assert model_config.model_name == "add"
    assert model_config.max_batch_size == 16
    assert model_config.batching is True

    assert len(model_config.instance_group) == 1

    device_kind = list(model_config.instance_group.keys())[0]
    device_count = list(model_config.instance_group.values())[0]
    assert device_kind.value == DeviceKind.KIND_CPU.value
    assert device_count is None

    assert not model_config.decoupled

    assert model_config.backend_parameters == {"workspace-path": "/tmp"}
    assert model_config.inputs == [
        TensorSpec(name="INPUT_0", dtype=np.int32, shape=(-1,)),
        TensorSpec(name="INPUT_1", dtype=np.int32, shape=(-1,)),
    ]
    assert model_config.outputs == [
        TensorSpec(name="OUTPUT_0", dtype=np.int32, shape=(-1,)),
    ]
    assert model_config.batcher.preferred_batch_size == [64, 128]
    assert model_config.batcher.max_queue_delay_microseconds == 100
    assert model_config.batcher.preserve_ordering is True

    assert model_config.batcher.priority_levels == 2
    assert model_config.batcher.default_priority_level == 1

    assert model_config.batcher.default_queue_policy is not None
    assert model_config.batcher.default_queue_policy.allow_timeout_override is True
    assert model_config.batcher.default_queue_policy.default_timeout_microseconds == 100
    assert model_config.batcher.default_queue_policy.max_queue_size == 10
    assert model_config.batcher.default_queue_policy.timeout_action == TimeoutAction.DELAY

    assert model_config.batcher.priority_queue_policy is not None
    assert len(model_config.batcher.priority_queue_policy) == 2

    priority_queue_policy1 = model_config.batcher.priority_queue_policy[1]
    assert priority_queue_policy1.allow_timeout_override is True
    assert priority_queue_policy1.default_timeout_microseconds == 100
    assert priority_queue_policy1.max_queue_size == 10
    assert priority_queue_policy1.timeout_action == TimeoutAction.DELAY

    priority_queue_policy2 = model_config.batcher.priority_queue_policy[2]
    assert priority_queue_policy2.allow_timeout_override is False
    assert priority_queue_policy2.default_timeout_microseconds == 1000
    assert priority_queue_policy2.max_queue_size == 2
    assert priority_queue_policy2.timeout_action == TimeoutAction.REJECT


def test_parse_from_dict_return_model_config_when_response_cache_config_used():
    model_config = ModelConfigParser.from_dict(model_config_dict=minimal_response_model_config)
    assert model_config.model_name == "minimal"
    assert model_config.max_batch_size == 0
    assert model_config.batching is False

    assert len(model_config.instance_group) == 1

    device_kind = list(model_config.instance_group.keys())[0]
    device_count = list(model_config.instance_group.values())[0]
    assert device_kind.value == DeviceKind.KIND_CPU.value
    assert device_count is None

    assert not model_config.decoupled

    assert model_config.backend_parameters == {"workspace-path": "/tmp"}
    assert model_config.inputs[0] == TensorSpec(name="INPUT_0", dtype=np.int32, shape=(-1,))
    assert model_config.outputs[0] == TensorSpec(name="OUTPUT_0", dtype=np.int32, shape=(-1,))

    assert model_config.batcher is None

    assert model_config.response_cache == ResponseCache(enable=True)


def test_parse_from_file_raise_error_when_file_with_invalid_model_config_passed():
    config_path = pathlib.Path(__file__).parent.resolve() / "assets" / "invalid_config.pbtxt"
    with pytest.raises(PyTritonModelConfigError):
        ModelConfigParser.from_file(config_path=config_path)


def test_parse_from_file_return_model_config_when_file_with_valid_model_config_passed():
    config_path = pathlib.Path(__file__).parent.resolve() / "assets" / "valid_config.pbtxt"
    model_config = ModelConfigParser.from_file(config_path=config_path)

    assert model_config.model_name == "simple"
    assert model_config.backend == "python"

    assert model_config.max_batch_size == 8

    assert model_config.inputs == [
        TensorSpec(name="INPUT0", dtype=np.int32, shape=(16,)),
        TensorSpec(name="INPUT1", dtype=np.int32, shape=(16,)),
    ]
    assert model_config.outputs == [
        TensorSpec(name="OUTPUT0", dtype=np.int32, shape=(16,)),
        TensorSpec(name="OUTPUT1", dtype=np.int32, shape=(16,)),
    ]


def test_parse_from_file_return_model_config_when_file_with_full_supported_config_passed():
    config_path = pathlib.Path(__file__).parent.resolve() / "assets" / "full_config.pbtxt"
    model_config = ModelConfigParser.from_file(config_path=config_path)

    assert model_config == full_model_config
