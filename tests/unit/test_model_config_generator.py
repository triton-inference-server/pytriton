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
import pytest

from pytriton.exceptions import PyTritonBadParameterError
from pytriton.model_config.common import DeviceKind, DynamicBatcher, QueuePolicy, TimeoutAction
from pytriton.model_config.generator import ModelConfigGenerator
from pytriton.model_config.triton_model_config import ResponseCache, TensorSpec, TritonModelConfig

from .common import full_model_config


def _load_config(config_path: pathlib.Path):
    """Load model config from path.

    Args:
        config_path: path to file with model config

    Returns:
        Dictionary with configuration
    """
    from google.protobuf import json_format, text_format  # pytype: disable=pyi-error
    from tritonclient.grpc import model_config_pb2  # pytype: disable=import-error

    with config_path.open("r") as config_file:
        payload = config_file.read()
        model_config_proto = text_format.Parse(payload, model_config_pb2.ModelConfig())

    model_config_dict = json_format.MessageToDict(model_config_proto, preserving_proto_field_name=True)
    return model_config_dict


def test_set_batching_raise_error_when_mbs_is_0_and_batching_is_not_disabled():
    model_config = TritonModelConfig(model_name="simple", batching=True, max_batch_size=0)
    generator = ModelConfigGenerator(model_config)

    model_config_data = {}
    with pytest.raises(PyTritonBadParameterError, match="The `max_batch_size` must be greater or equal to 1."):
        generator._set_batching(model_config_data)


def test_set_batching_raise_error_when_mbs_is_less_then_0_and_batching_is_not_disabled():
    model_config = TritonModelConfig(model_name="simple", batching=True, max_batch_size=-1)
    generator = ModelConfigGenerator(model_config)

    model_config_data = {}
    with pytest.raises(PyTritonBadParameterError, match="The `max_batch_size` must be greater or equal to 1."):
        generator._set_batching(model_config_data)


def test_set_batching_set_max_batch_size_to_0_when_batching_is_disabled():
    model_config = TritonModelConfig(model_name="simple", batching=False)
    generator = ModelConfigGenerator(model_config)

    model_config_data = {}
    generator._set_batching(model_config_data)

    assert model_config_data == {
        "max_batch_size": 0,
    }


def test_set_batching_set_max_batch_size_to_default_when_batching_set_to_default():
    model_config = TritonModelConfig(model_name="simple", batching=True)
    generator = ModelConfigGenerator(model_config)

    model_config_data = {}
    generator._set_batching(model_config_data)

    assert model_config_data == {"max_batch_size": 4}


def test_set_batching_set_dynamic_batching_field_when_batcher_set_to_dynamic():
    model_config = TritonModelConfig(model_name="simple", batching=True, batcher=DynamicBatcher())
    generator = ModelConfigGenerator(model_config)

    model_config_data = {}
    generator._set_batching(model_config_data)

    assert model_config_data == {"max_batch_size": 4, "dynamic_batching": {}}


def test_set_batching_set_max_batch_size_when_batching_enabled_and_value_passed():
    model_config = TritonModelConfig(model_name="simple", batching=True, max_batch_size=16)
    generator = ModelConfigGenerator(model_config)

    model_config_data = {}
    generator._set_batching(model_config_data)

    assert model_config_data == {"max_batch_size": 16}


def test_set_batching_set_dynamic_batching_config_when_dynamic_batching_enabled_and_flags_passed():
    model_config = TritonModelConfig(
        model_name="simple",
        batching=True,
        max_batch_size=16,
        batcher=DynamicBatcher(
            preferred_batch_size=[16, 32],
            max_queue_delay_microseconds=100,
            preserve_ordering=True,
        ),
    )
    generator = ModelConfigGenerator(model_config)

    model_config_data = {}
    generator._set_batching(model_config_data)

    assert model_config_data["max_batch_size"] == 16
    assert model_config_data["dynamic_batching"] == {
        "preferredBatchSize": [16, 32],
        "maxQueueDelayMicroseconds": 100,
        "preserveOrdering": True,
    }

    model_config_data = {}
    model_config.batcher.preserve_ordering = False

    generator._set_batching(model_config_data)

    assert model_config_data == {
        "max_batch_size": 16,
        "dynamic_batching": {
            "preferredBatchSize": [16, 32],
            "maxQueueDelayMicroseconds": 100,
        },
    }


def test_set_batching_raise_exception_when_invalid_default_priority_level_passed():
    model_config = TritonModelConfig(
        model_name="simple",
        batching=True,
        max_batch_size=16,
        batcher=DynamicBatcher(
            priority_levels=5,
            default_priority_level=6,
        ),
    )
    generator = ModelConfigGenerator(model_config)

    model_config_data = {}

    with pytest.raises(PyTritonBadParameterError, match="The `default_priority_level` must be between 1 and 5."):
        generator._set_batching(model_config_data)


def test_set_batching_set_dynamic_batching_config_when_default_queue_policy_passed():
    model_config = TritonModelConfig(
        model_name="simple",
        batching=True,
        max_batch_size=16,
        batcher=DynamicBatcher(
            default_queue_policy=QueuePolicy(
                allow_timeout_override=True,
                timeout_action=TimeoutAction.DELAY,
                default_timeout_microseconds=100,
                max_queue_size=2,
            )
        ),
    )
    generator = ModelConfigGenerator(model_config)

    model_config_data = {}
    generator._set_batching(model_config_data)

    assert model_config_data == {
        "max_batch_size": 16,
        "dynamic_batching": {
            "defaultQueuePolicy": {
                "allowTimeoutOverride": 1,
                "timeoutAction": "DELAY",
                "defaultTimeoutMicroseconds": 100,
                "maxQueueSize": 2,
            }
        },
    }


def test_set_batching_raise_exception_when_priority_queue_policy_passed_but_no_default_priority_level():
    model_config = TritonModelConfig(
        model_name="simple",
        batching=True,
        max_batch_size=16,
        batcher=DynamicBatcher(
            priority_queue_policy={
                1: QueuePolicy(
                    allow_timeout_override=True,
                    timeout_action=TimeoutAction.DELAY,
                    default_timeout_microseconds=100,
                    max_queue_size=2,
                )
            },
        ),
    )
    generator = ModelConfigGenerator(model_config)

    model_config_data = {}

    with pytest.raises(
        PyTritonBadParameterError,
        match="Provide the `priority_levels` if you want to define `priority_queue_policy` for Dynamic Batching.",
    ):
        generator._set_batching(model_config_data)


def test_set_batching_raise_exception_when_invalid_priority_queue_policy_passed():
    model_config = TritonModelConfig(
        model_name="simple",
        batching=True,
        max_batch_size=16,
        batcher=DynamicBatcher(
            priority_levels=5,
            default_priority_level=2,
            priority_queue_policy={
                6: QueuePolicy(
                    allow_timeout_override=True,
                    timeout_action=TimeoutAction.DELAY,
                    default_timeout_microseconds=100,
                    max_queue_size=2,
                )
            },
        ),
    )
    generator = ModelConfigGenerator(model_config)

    model_config_data = {}

    with pytest.raises(
        PyTritonBadParameterError, match="Invalid `priority`=6 provided. The value must be between 1 and 5."
    ):
        generator._set_batching(model_config_data)


def test_set_batching_set_dynamic_batching_config_when_priority_queue_policy_passed():
    model_config = TritonModelConfig(
        model_name="simple",
        batching=True,
        max_batch_size=16,
        batcher=DynamicBatcher(
            priority_levels=3,
            default_priority_level=1,
            priority_queue_policy={
                2: QueuePolicy(
                    allow_timeout_override=True,
                    timeout_action=TimeoutAction.DELAY,
                    default_timeout_microseconds=100,
                    max_queue_size=2,
                )
            },
        ),
    )
    generator = ModelConfigGenerator(model_config)

    model_config_data = {}
    generator._set_batching(model_config_data)

    assert model_config_data == {
        "max_batch_size": 16,
        "dynamic_batching": {
            "priorityLevels": 3,
            "defaultPriorityLevel": 1,
            "priorityQueuePolicy": {
                2: {
                    "allowTimeoutOverride": 1,
                    "timeoutAction": "DELAY",
                    "defaultTimeoutMicroseconds": 100,
                    "maxQueueSize": 2,
                }
            },
        },
    }


def test_set_instance_group_not_update_data_when_instance_group_not_provided():
    model_config = TritonModelConfig(model_name="simple")
    generator = ModelConfigGenerator(model_config)

    model_config_data = {}
    generator._set_instance_group(model_config_data)

    assert model_config_data == {}


def test_set_instance_group_set_instance_configuration_when_single_config_provided():
    model_config = TritonModelConfig(model_name="simple", instance_group={DeviceKind.KIND_GPU: None})
    generator = ModelConfigGenerator(model_config)

    model_config_data = {}
    generator._set_instance_group(model_config_data)

    assert model_config_data == {
        "instance_group": [
            {"kind": DeviceKind.KIND_GPU.value, "count": None},
        ]
    }


def test_set_instance_group_set_instance_configuration_when_single_multiple_configs_provided():
    model_config = TritonModelConfig(
        model_name="simple",
        instance_group={
            DeviceKind.KIND_GPU: None,
            DeviceKind.KIND_CPU: 10,
        },
    )
    generator = ModelConfigGenerator(model_config)

    model_config_data = {}
    generator._set_instance_group(model_config_data)

    assert model_config_data == {
        "instance_group": [
            {"kind": DeviceKind.KIND_GPU.value, "count": None},
            {"kind": DeviceKind.KIND_CPU.value, "count": 10},
        ]
    }


def test_transaction_policy_not_update_data_when_decoupled_execution_disabled():
    model_config = TritonModelConfig(model_name="simple", decoupled=False)
    generator = ModelConfigGenerator(model_config)
    model_config_data = {}
    generator._set_model_transaction_policy(model_config_data)
    assert model_config_data == {}


def test_transaction_policy_added_when_decoupled_execution_enabled():
    model_config = TritonModelConfig(model_name="simple", decoupled=True)
    generator = ModelConfigGenerator(model_config)
    model_config_data = {}
    generator._set_model_transaction_policy(model_config_data)
    assert model_config_data == {"model_transaction_policy": {"decoupled": True}}


def test_set_backend_parameters_not_update_data_when_parameters_not_provided():
    model_config = TritonModelConfig(model_name="simple")
    generator = ModelConfigGenerator(model_config)

    model_config_data = {}
    generator._set_backend_parameters(model_config_data)

    assert model_config_data == {}


def test_set_backend_parameters_update_config_when_parameters_provided():
    model_config = TritonModelConfig(
        model_name="simple",
        backend_parameters={
            "parameter1": "value1",
            "parameter2": "value2",
        },
    )
    generator = ModelConfigGenerator(model_config)

    model_config_data = {}
    generator._set_backend_parameters(model_config_data)

    assert model_config_data == {
        "parameters": {
            "parameter1": {"string_value": "value1"},
            "parameter2": {"string_value": "value2"},
        }
    }


def test_set_model_signature_not_update_data_when_spec_not_provided():
    model_config = TritonModelConfig(model_name="simple")
    generator = ModelConfigGenerator(model_config)

    model_config_data = {}
    generator._set_model_signature(model_config_data)

    assert model_config_data == {}


def test_set_model_signature_update_data_when_spec_provided():
    model_config = TritonModelConfig(
        model_name="simple",
        inputs=[
            TensorSpec(name="INPUT_1", dtype=np.float32, shape=(-1,)),
            TensorSpec(name="INPUT_2", dtype=object, shape=(-1,)),
        ],
        outputs=[
            TensorSpec(name="OUTPUT_1", dtype=np.int32, shape=(1000,)),
        ],
    )
    generator = ModelConfigGenerator(model_config)

    model_config_data = {}
    generator._set_model_signature(model_config_data)

    assert model_config_data == {
        "input": [
            {"name": "INPUT_1", "data_type": "TYPE_FP32", "dims": [-1]},
            {"name": "INPUT_2", "data_type": "TYPE_STRING", "dims": [-1]},
        ],
        "output": [
            {"name": "OUTPUT_1", "data_type": "TYPE_INT32", "dims": [1000]},
        ],
    }


def test_set_model_signature_raise_error_when_output_marked_as_optional():
    model_config = TritonModelConfig(
        model_name="simple",
        inputs=[
            TensorSpec(name="INPUT_1", dtype=np.float32, shape=(-1,)),
            TensorSpec(name="INPUT_2", dtype=object, shape=(-1,)),
        ],
        outputs=[
            TensorSpec(name="OUTPUT_1", dtype=np.int32, shape=(1000,), optional=True),
        ],
    )
    generator = ModelConfigGenerator(model_config)

    model_config_data = {}
    with pytest.raises(
        PyTritonBadParameterError,
        match="Optional flag for outputs is not supported. Outputs marked as optional: OUTPUT_1.",
    ):
        generator._set_model_signature(model_config_data)


def test_get_config_call_config_generator_methods(mocker):
    mock_set_batching = mocker.patch.object(ModelConfigGenerator, "_set_batching")
    mock_set_model_signature = mocker.patch.object(ModelConfigGenerator, "_set_model_signature")
    mock_set_instance_group = mocker.patch.object(ModelConfigGenerator, "_set_instance_group")
    mock_set_backend_parameters = mocker.patch.object(ModelConfigGenerator, "_set_backend_parameters")

    model_config = TritonModelConfig(model_name="simple")

    generator = ModelConfigGenerator(model_config)
    model_config_data = generator.get_config()

    assert model_config_data == {
        "name": "simple",
        "backend": "python",
    }

    assert mock_set_batching.called is True
    assert mock_set_model_signature.called is True
    assert mock_set_instance_group.called is True
    assert mock_set_backend_parameters.called is True


def test_get_config_return_defaults_when_minimal_config_passed():
    model_config = TritonModelConfig(model_name="simple")

    generator = ModelConfigGenerator(model_config)
    model_config_data = generator.get_config()

    assert model_config_data == {
        "name": "simple",
        "backend": "python",
        "max_batch_size": 4,
    }


def test_get_config_return_response_cache_when_enabled_for_model():
    model_config = TritonModelConfig(model_name="simple", response_cache=ResponseCache(enable=True))

    generator = ModelConfigGenerator(model_config)
    model_config_data = generator.get_config()

    assert model_config_data == {
        "name": "simple",
        "backend": "python",
        "max_batch_size": 4,
        "response_cache": {"enable": True},
    }


def test_get_config_return_response_cache_when_disabled_for_model():
    model_config = TritonModelConfig(model_name="simple", response_cache=ResponseCache(enable=False))

    generator = ModelConfigGenerator(model_config)
    model_config_data = generator.get_config()

    assert model_config_data == {
        "name": "simple",
        "backend": "python",
        "max_batch_size": 4,
        "response_cache": {"enable": False},
    }


def test_to_file_save_config_to_file_and_override_max_batch_size_when_batching_disabled(mocker):
    mock_get_config = mocker.patch.object(ModelConfigGenerator, "get_config")
    mock_get_config.return_value = {
        "name": "simple",
        "backend": "python",
        "max_batch_size": 0,
        "dynamic_batching": {},
    }

    model_config = TritonModelConfig(model_name="simple")
    generator = ModelConfigGenerator(model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "python",
            "dynamic_batching": {},
        }


def test_to_file_save_config_to_file_when_full_config_specified():
    generator = ModelConfigGenerator(full_model_config)

    with tempfile.NamedTemporaryFile() as fp:
        generator.to_file(fp.name)

        config_path = pathlib.Path(fp.name)
        assert config_path.exists() is True

        data = _load_config(config_path)

        assert data == {
            "name": "simple",
            "backend": "python",
            "max_batch_size": 16,
            "dynamic_batching": {
                "preferred_batch_size": [16, 32],
                "max_queue_delay_microseconds": "100",
                "preserve_ordering": True,
                "priority_levels": 3,
                "default_priority_level": 1,
                "default_queue_policy": {
                    "allow_timeout_override": True,
                    "timeout_action": "DELAY",
                    "default_timeout_microseconds": "100",
                    "max_queue_size": 2,
                },
                "priority_queue_policy": {
                    "2": {
                        "allow_timeout_override": True,
                        "timeout_action": "DELAY",
                        "default_timeout_microseconds": "100",
                        "max_queue_size": 3,
                    }
                },
            },
            "instance_group": [
                {
                    "count": 1,
                    "kind": "KIND_CPU",
                },
                {
                    "count": 2,
                    "kind": "KIND_GPU",
                },
            ],
            "input": [
                {"name": "INPUT_1", "data_type": "TYPE_FP32", "dims": ["-1"]},
                {"name": "INPUT_2", "data_type": "TYPE_STRING", "dims": ["-1"]},
            ],
            "output": [
                {"name": "OUTPUT_1", "data_type": "TYPE_INT32", "dims": ["1000"]},
            ],
            "parameters": {
                "parameter1": {"string_value": "value1"},
                "parameter2": {"string_value": "value2"},
            },
            "response_cache": {"enable": True},
            "model_transaction_policy": {"decoupled": True},
        }
