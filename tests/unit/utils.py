# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import typing

import numpy as np
import tritonclient.grpc
import tritonclient.http
import tritonclient.utils

from pytriton.model_config.generator import ModelConfigGenerator
from pytriton.model_config.triton_model_config import TritonModelConfig


def verify_equalness_of_dicts_with_ndarray(a_dict, b_dict):
    assert a_dict.keys() == b_dict.keys(), f"{a_dict} != {b_dict}"
    for output_name in a_dict:
        assert isinstance(
            a_dict[output_name], type(b_dict[output_name])
        ), f"type(a[{output_name}])={type(a_dict[output_name])} != type(b[{output_name}])={type(b_dict[output_name])}"
        if isinstance(a_dict[output_name], np.ndarray):
            assert a_dict[output_name].dtype == b_dict[output_name].dtype
            assert a_dict[output_name].shape == b_dict[output_name].shape
            if np.issubdtype(a_dict[output_name].dtype, np.number):
                assert np.allclose(b_dict[output_name], a_dict[output_name])
            else:
                assert np.array_equal(b_dict[output_name], a_dict[output_name])
        else:
            assert a_dict[output_name] == b_dict[output_name]


def wrap_to_grpc_infer_result(
    model_config: TritonModelConfig, request_id: str, outputs_dict: typing.Dict[str, np.ndarray]
):
    raw_output_contents = [output_data.tobytes() for output_data in outputs_dict.values()]
    return tritonclient.grpc.InferResult(
        tritonclient.grpc.service_pb2.ModelInferResponse(
            model_name=model_config.model_name,
            model_version=str(model_config.model_version),
            id=request_id,
            outputs=[
                tritonclient.grpc.service_pb2.ModelInferResponse.InferOutputTensor(
                    name=output_name,
                    datatype=tritonclient.utils.np_to_triton_dtype(output_data.dtype),
                    shape=output_data.shape,
                )
                for output_name, output_data in outputs_dict.items()
            ],
            raw_output_contents=raw_output_contents,
        )
    )


def wrap_to_http_infer_result(
    model_config: TritonModelConfig, request_id: str, outputs_dict: typing.Dict[str, np.ndarray]
):
    raw_output_contents = [output_data.tobytes() for output_data in outputs_dict.values()]
    buffer = b"".join(raw_output_contents)

    content = {
        "outputs": [
            {
                "name": name,
                "datatype": tritonclient.utils.np_to_triton_dtype(output_data.dtype),
                "shape": list(output_data.shape),
                "parameters": {"binary_data_size": len(output_data.tobytes())},
            }
            for name, output_data in outputs_dict.items()
        ]
    }
    header = json.dumps(content).encode("utf-8")
    response_body = header + buffer

    return tritonclient.http.InferResult.from_response_body(response_body, False, len(header))


def extract_array_from_grpc_infer_input(input_: tritonclient.grpc.InferInput):
    np_array = np.frombuffer(input_._raw_content, dtype=tritonclient.utils.triton_to_np_dtype(input_.datatype()))
    np_array = np_array.reshape(input_.shape())
    return np_array


def extract_array_from_http_infer_input(input_: tritonclient.http.InferInput):
    np_array = np.frombuffer(input_._raw_data, dtype=tritonclient.utils.triton_to_np_dtype(input_.datatype()))
    np_array = np_array.reshape(input_.shape())
    return np_array


def patch_grpc_client__server_up_and_ready(mocker):
    mocker.patch.object(tritonclient.grpc.InferenceServerClient, "is_server_ready").return_value = True
    mocker.patch.object(tritonclient.grpc.InferenceServerClient, "is_server_live").return_value = True


def patch_http_client__server_up_and_ready(mocker):
    mocker.patch.object(tritonclient.http.InferenceServerClient, "is_server_ready").return_value = True
    mocker.patch.object(tritonclient.http.InferenceServerClient, "is_server_live").return_value = True


def patch_grpc_client__model_up_and_ready(mocker, model_config: TritonModelConfig):
    from google.protobuf import json_format  # pytype: disable=pyi-error
    from tritonclient.grpc import model_config_pb2, service_pb2  # pytype: disable=pyi-error

    mock_get_repo_index = mocker.patch.object(tritonclient.grpc.InferenceServerClient, "get_model_repository_index")
    mock_get_repo_index.return_value = service_pb2.RepositoryIndexResponse(
        models=[
            service_pb2.RepositoryIndexResponse.ModelIndex(
                name=model_config.model_name, version="1", state="READY", reason=""
            ),
        ]
    )

    mocker.patch.object(tritonclient.grpc.InferenceServerClient, "is_model_ready").return_value = True

    model_config_dict = ModelConfigGenerator(model_config).get_config()
    model_config_protobuf = json_format.ParseDict(model_config_dict, model_config_pb2.ModelConfig())
    response = service_pb2.ModelConfigResponse(config=model_config_protobuf)
    response_dict = json.loads(json_format.MessageToJson(response, preserving_proto_field_name=True))
    mock_get_model_config = mocker.patch.object(tritonclient.grpc.InferenceServerClient, "get_model_config")
    mock_get_model_config.return_value = response_dict


def patch_http_client__model_up_and_ready(mocker, model_config: TritonModelConfig):
    mock_get_repo_index = mocker.patch.object(tritonclient.http.InferenceServerClient, "get_model_repository_index")
    mock_get_repo_index.return_value = [
        {"name": model_config.model_name, "version": "1", "state": "READY", "reason": ""}
    ]

    mocker.patch.object(tritonclient.http.InferenceServerClient, "is_model_ready").return_value = True

    model_config_dict = ModelConfigGenerator(model_config).get_config()
    mock_get_model_config = mocker.patch.object(tritonclient.http.InferenceServerClient, "get_model_config")
    mock_get_model_config.return_value = model_config_dict
