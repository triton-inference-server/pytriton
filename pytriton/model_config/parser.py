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
"""ModelConfigParser class definition.

Provide functionality to parse the Triton model configuration stored in file or form of dictionary into the object of
class ModelConfig.

    Examples of use:

        # Parse from dict
        model_config = ModelConfigParser.from_dict(model_config_dict)

        # Parse from file
        model_config = ModelConfigParser.from_file("/path/to/config.pbtxt")

"""
import json
import logging
import pathlib
from typing import Dict

import numpy as np
from google.protobuf import json_format, text_format  # pytype: disable=pyi-error

from pytriton.exceptions import PyTritonModelConfigError

from .common import QueuePolicy, TimeoutAction
from .triton_model_config import DeviceKind, DynamicBatcher, ResponseCache, TensorSpec, TritonModelConfig

try:
    import tritonclient.grpc as grpc_client
    from tritonclient import utils as client_utils  # noqa: F401
except ImportError:
    try:
        import tritonclientutils as client_utils  # noqa: F401
        import tritongrpcclient as grpc_client
    except ImportError:
        client_utils = None
        grpc_client = None

LOGGER = logging.getLogger(__name__)


class ModelConfigParser:
    """Provide functionality to parse dictionary or file to ModelConfig object."""

    @classmethod
    def from_dict(cls, model_config_dict: Dict) -> TritonModelConfig:
        """Create ModelConfig from configuration stored in dictionary.

        Args:
            model_config_dict: Dictionary with model config

        Returns:
            A ModelConfig object with data parsed from the dictionary
        """
        LOGGER.debug(f"Parsing Triton config model from dict: \n{json.dumps(model_config_dict, indent=4)}")

        if model_config_dict.get("max_batch_size", 0) > 0:
            batching = True
        else:
            batching = False

        dynamic_batcher_config = model_config_dict.get("dynamic_batching")
        if dynamic_batcher_config is not None:
            batcher = cls._parse_dynamic_batching(dynamic_batcher_config)
        else:
            batcher = None

        instance_group = {
            DeviceKind(entry["kind"]): entry.get("count") for entry in model_config_dict.get("instance_group", [])
        }

        decoupled = model_config_dict.get("model_transaction_policy", {}).get("decoupled", False)

        backend_parameters_config = model_config_dict.get("parameters", [])
        if isinstance(backend_parameters_config, list):
            # If the backend_parameters_config is a list of strings, use them as keys with empty values
            LOGGER.debug(f"backend_parameters_config is a list of strings: {backend_parameters_config}")
            backend_parameters = {name: "" for name in backend_parameters_config}
        elif isinstance(backend_parameters_config, dict):
            # If the backend_parameters_config is a dictionary, use the key and "string_value" fields as key-value pairs
            LOGGER.debug(f"backend_parameters_config is a dictionary: {backend_parameters_config}")
            backend_parameters = {
                name: backend_parameters_config[name]["string_value"] for name in backend_parameters_config
            }
        else:
            # Otherwise, raise an error
            LOGGER.error(
                f"Invalid type {type(backend_parameters_config)} for backend_parameters_config: {backend_parameters_config}"
            )
            raise TypeError(f"Invalid type for backend_parameters_config: {type(backend_parameters_config)}")

        inputs = [
            cls.rewrite_io_spec(item, "input", idx) for idx, item in enumerate(model_config_dict.get("input", []))
        ] or None
        outputs = [
            cls.rewrite_io_spec(item, "output", idx) for idx, item in enumerate(model_config_dict.get("output", []))
        ] or None

        response_cache_config = model_config_dict.get("response_cache")
        if response_cache_config:
            response_cache = cls._parse_response_cache(response_cache_config)
        else:
            response_cache = None

        return TritonModelConfig(
            model_name=model_config_dict["name"],
            batching=batching,
            max_batch_size=model_config_dict.get("max_batch_size", 0),
            batcher=batcher,
            inputs=inputs,
            outputs=outputs,
            instance_group=instance_group,
            decoupled=decoupled,
            backend_parameters=backend_parameters,
            response_cache=response_cache,
        )

    @classmethod
    def from_file(cls, *, config_path: pathlib.Path) -> TritonModelConfig:
        """Create ModelConfig from configuration stored in file.

        Args:
            config_path: location of file with model config

        Returns:
            A ModelConfig object with data parsed from the file
        """
        from tritonclient.grpc import model_config_pb2  # pytype: disable=import-error

        LOGGER.debug(f"Parsing Triton config model config_path={config_path}")

        with config_path.open("r") as config_file:
            payload = config_file.read()
            model_config_proto = text_format.Parse(payload, model_config_pb2.ModelConfig())

        model_config_dict = json_format.MessageToDict(model_config_proto, preserving_proto_field_name=True)
        return ModelConfigParser.from_dict(model_config_dict=model_config_dict)

    @classmethod
    def rewrite_io_spec(cls, item: Dict, io_type: str, idx: int) -> TensorSpec:
        """Rewrite the IO Spec provided in form of dictionary to TensorSpec.

        Args:
            item: IO data for input
            io_type: Type of the IO (input or output)
            idx: Index of IO

        Returns:
            TensorSpec with input or output data
        """
        name = item.get("name")
        if not name:
            raise PyTritonModelConfigError(f"Name for {io_type} at index {idx} not provided.")

        data_type = item.get("data_type")
        if not data_type:
            raise PyTritonModelConfigError(f"Data type for {io_type} with name `{name}` not defined.")

        data_type_val = data_type.split("_")
        if len(data_type_val) != 2:
            raise PyTritonModelConfigError(
                f"Invalid data type `{data_type}` for {io_type} with name `{name}` not defined. "
                "The expected name is TYPE_{type}."
            )

        data_type = data_type_val[1]
        if data_type == "STRING":
            dtype = np.bytes_
        else:
            dtype = client_utils.triton_to_np_dtype(data_type)
            if dtype is None:
                raise PyTritonModelConfigError(f"Unsupported data type `{data_type}` for {io_type} with name `{name}`")

            dtype = np.dtype("bool") if dtype == bool else dtype

        dims = item.get("dims", [])
        if not dims:
            raise PyTritonModelConfigError(f"Dimension for {io_type} with name `{name}` not defined.")

        shape = tuple(int(s) for s in dims)

        optional = item.get("optional", False)
        return TensorSpec(name=item["name"], shape=shape, dtype=dtype, optional=optional)

    @classmethod
    def _parse_dynamic_batching(cls, dynamic_batching_config: Dict) -> DynamicBatcher:
        """Parse config to create DynamicBatcher object.

        Args:
            dynamic_batching_config: Configuration of dynamic batcher from config

        Returns:
            DynamicBatcher object with configuration
        """
        default_queue_policy = None
        default_queue_policy_config = dynamic_batching_config.get("default_queue_policy")
        if default_queue_policy_config:
            default_queue_policy = QueuePolicy(
                timeout_action=TimeoutAction(
                    default_queue_policy_config.get("timeout_action", TimeoutAction.REJECT.value)
                ),
                default_timeout_microseconds=int(default_queue_policy_config.get("default_timeout_microseconds", 0)),
                allow_timeout_override=bool(default_queue_policy_config.get("allow_timeout_override", False)),
                max_queue_size=int(default_queue_policy_config.get("max_queue_size", 0)),
            )

        priority_queue_policy = None
        priority_queue_policy_config = dynamic_batching_config.get("priority_queue_policy")
        if priority_queue_policy_config:
            priority_queue_policy = {}
            for priority, queue_policy_config in priority_queue_policy_config.items():
                queue_policy = QueuePolicy(
                    timeout_action=TimeoutAction(queue_policy_config.get("timeout_action", TimeoutAction.REJECT.value)),
                    default_timeout_microseconds=int(queue_policy_config.get("default_timeout_microseconds", 0)),
                    allow_timeout_override=bool(queue_policy_config.get("allow_timeout_override", False)),
                    max_queue_size=int(queue_policy_config.get("max_queue_size", 0)),
                )
                priority_queue_policy[int(priority)] = queue_policy

        batcher = DynamicBatcher(
            preferred_batch_size=dynamic_batching_config.get("preferred_batch_size"),
            max_queue_delay_microseconds=int(dynamic_batching_config.get("max_queue_delay_microseconds", 0)),
            preserve_ordering=bool(dynamic_batching_config.get("preserve_ordering", False)),
            priority_levels=int(dynamic_batching_config.get("priority_levels", 0)),
            default_priority_level=int(dynamic_batching_config.get("default_priority_level", 0)),
            default_queue_policy=default_queue_policy,
            priority_queue_policy=priority_queue_policy,
        )
        return batcher

    @classmethod
    def _parse_response_cache(cls, response_cache_config: Dict) -> ResponseCache:
        """Parse config for response cache.

        Args:
            response_cache_config: response cache configuration

        Returns:
            ResponseCache object with configuration
        """
        response_cache = ResponseCache(
            enable=bool(response_cache_config["enable"]),
        )
        return response_cache
