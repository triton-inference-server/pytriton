# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import logging

import numpy as np
import pytest

from pytriton.model_config.triton_model_config import TensorSpec, TritonModelConfig
from pytriton.proxy.validators import validate_output_data, validate_output_dtype_and_shape, validate_outputs

LOGGER = logging.getLogger("tests.unit.test_proxy_validators")
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

MY_MODEL_OUTPUTS = {output.name: output for output in MY_MODEL_CONFIG.outputs}


def test_validate_outputs_throws_exception_when_outputs_is_not_a_list():
    outputs = {"output1": np.array([1, 2, 3]), "output2": np.array([1, 2, 3])}

    with pytest.raises(
        ValueError,
        match=r"Outputs returned by `Foo` model callable must be list of response dicts with numpy arrays",
    ):
        validate_outputs(
            model_config=MY_MODEL_CONFIG,
            model_outputs=MY_MODEL_OUTPUTS,
            outputs=outputs,
            strict=False,
            requests_number=1,
        )


def test_validate_outputs_throws_exception_when_outputs_number_is_not_equal_to_requests_number():
    outputs = [{"output1": np.array([1, 2, 3]), "output2": np.array([1, 2, 3])}]

    with pytest.raises(
        ValueError,
        match=r"Number of outputs returned by `Foo` inference callable "
        r"\(1\) does not match number of requests \(2\) received from Triton\.",
    ):
        validate_outputs(
            model_config=MY_MODEL_CONFIG,
            model_outputs=MY_MODEL_OUTPUTS,
            outputs=outputs,
            strict=False,
            requests_number=2,
        )


def test_validate_outputs_throws_exception_when_outputs_is_not_a_list_of_dicts():
    outputs = [np.array([1, 2, 3]), np.array([1, 2, 3])]

    with pytest.raises(
        ValueError,
        match=r"Outputs returned by `Foo` model callable must be list of response dicts with numpy arrays",
    ):
        validate_outputs(
            model_config=MY_MODEL_CONFIG,
            model_outputs=MY_MODEL_OUTPUTS,
            outputs=outputs,
            strict=False,
            requests_number=len(outputs),
        )


def test_validate_outputs_call_validate_outputs_data_if_strict_is_false(mocker):
    outputs = [{"output1": np.array([1, 2, 3]), "output2": np.array([1, 2, 3])}]
    mock_validate_outputs_data = mocker.patch("pytriton.proxy.validators.validate_output_data")
    mock_validate_output_dtype_and_shape = mocker.patch("pytriton.proxy.validators.validate_output_dtype_and_shape")

    validate_outputs(
        model_config=MY_MODEL_CONFIG,
        model_outputs=MY_MODEL_OUTPUTS,
        outputs=outputs,
        strict=False,
        requests_number=len(outputs),
    )

    assert mock_validate_outputs_data.called is True
    assert mock_validate_output_dtype_and_shape.called is False


def test_validate_outputs_call_validate_outputs_data_if_strict_is_true(mocker):
    outputs = [{"output1": np.array([1, 2, 3]), "output2": np.array([1, 2, 3])}]
    mock_validate_outputs_data = mocker.patch("pytriton.proxy.validators.validate_output_data")
    mock_validate_output_dtype_and_shape = mocker.patch("pytriton.proxy.validators.validate_output_dtype_and_shape")

    validate_outputs(
        model_config=MY_MODEL_CONFIG,
        model_outputs=MY_MODEL_OUTPUTS,
        outputs=outputs,
        strict=True,
        requests_number=len(outputs),
    )

    assert mock_validate_outputs_data.called is True
    assert mock_validate_output_dtype_and_shape.called is True


def test_validate_output_data_throws_exception_when_name_is_not_a_string():
    name = 12
    value = [1.0, 2.0, 3.0]

    with pytest.raises(
        ValueError,
        match=r"Not all keys returned by `Foo` model callable are string",
    ):
        validate_output_data(model_config=MY_MODEL_CONFIG, name=name, value=value)


def test_validate_output_data_throws_exception_when_value_is_not_numpy_array():
    name = "output1"
    value = [1.0, 2.0, 3.0]

    with pytest.raises(
        ValueError,
        match=r"Not all values returned by `Foo` model callable are numpy arrays",
    ):
        validate_output_data(model_config=MY_MODEL_CONFIG, name=name, value=value)


def test_validate_output_data_throws_exception_when_value_is_not_supported_data_type():
    name = "output1"
    value = np.array(["2000-01-01T12:00:00.000", "2002-01-01T12:00:00.000"], dtype="datetime64[ms]")

    with pytest.raises(
        ValueError,
        match=r"Only bool, numeric, string, unicode and object arrays "
        r"are supported by Triton \(dtype\.kind: biufOSU\)\. "
        "Returned `output1` for model `Foo` "
        r"has `M` dtype\.kind\.",
    ):
        validate_output_data(model_config=MY_MODEL_CONFIG, name=name, value=value)


def test_validate_output_data_throws_exception_when_value_is_list_of_strings():
    name = "output1"
    value = np.array(["abcd", "efgg"], dtype=np.object_)

    with pytest.raises(
        ValueError,
        match=r"Use string/byte-string instead of object for passing string in NumPy array from model `Foo`\.",
    ):
        validate_output_data(model_config=MY_MODEL_CONFIG, name=name, value=value)


def test_validate_output_data_throws_exception_when_value_is_list_of_ints_defined_as_object():
    name = "output1"
    value = np.array([123, 456], dtype=np.object_)

    with pytest.raises(
        ValueError,
        match=r"Only bytes as objects dtype are supported by PyTriton\. "
        "Returned `output1` from `Foo` "
        r"has `\<class 'int'\>` type\.",
    ):
        validate_output_data(model_config=MY_MODEL_CONFIG, name=name, value=value)


def test_validate_output_dtype_and_shape_throws_exception_when_name_not_in_model_config():
    name = "output3"
    value = np.array([[1.0, 2.0]], dtype=np.int32)

    with pytest.raises(
        ValueError,
        match=r"Returned output `output3` is not defined in model config for model `Foo`\.",
    ):
        validate_output_dtype_and_shape(
            model_config=MY_MODEL_CONFIG, model_outputs=MY_MODEL_OUTPUTS, name=name, value=value
        )


def test_validate_output_dtype_and_shape_throws_exception_when_value_has_incorrect_dtype_and_float_returned():
    name = "output1"
    value = np.array([[1.0], [2.0], [3.0]], dtype=float)

    with pytest.raises(
        ValueError,
        match=r"Returned output `output1` for model `Foo` has invalid type\. Returned: float64 \(f\). Expected: \<class 'numpy\.int32'\>\.",
    ):
        validate_output_dtype_and_shape(
            model_config=MY_MODEL_CONFIG, model_outputs=MY_MODEL_OUTPUTS, name=name, value=value
        )


def test_validate_output_dtype_and_shape_throws_exception_when_value_has_incorrect_dtype_and_bytes_returned():
    name = "output1"
    value = np.array([b"test1", b"test2", b"test3"], dtype=np.bytes_)

    with pytest.raises(
        ValueError,
        match=r"Returned output `output1` for model `Foo` has invalid type\. Returned: \|S5 \(S\). Expected: \<class 'numpy\.int32'\>\.",
    ):
        validate_output_dtype_and_shape(
            model_config=MY_MODEL_CONFIG, model_outputs=MY_MODEL_OUTPUTS, name=name, value=value
        )


def test_validate_output_dtype_and_shape_throws_exception_when_value_has_incorrect_shape():
    name = "output1"
    value = np.array([[[1], [2]], [[3], [4]]], dtype=np.int32)

    with pytest.raises(
        ValueError,
        match=r"Returned output `output1` for model `Foo` has invalid shapes\. Returned: \(2, 1\)\. Expected: \(3,\)\.",
    ):
        validate_output_dtype_and_shape(
            model_config=MY_MODEL_CONFIG, model_outputs=MY_MODEL_OUTPUTS, name=name, value=value
        )


def test_validate_output_dtype_and_shape_throws_exception_when_value_contains_too_little_items():
    name = "output1"
    value = np.array([[1.0, 2.0]], dtype=np.int32)

    with pytest.raises(
        ValueError,
        match=r"Returned output `output1` for model `Foo` has invalid shapes at one or more positions\. Returned: \(2,\)\. Expected: \(3,\)\.",
    ):
        validate_output_dtype_and_shape(
            model_config=MY_MODEL_CONFIG, model_outputs=MY_MODEL_OUTPUTS, name=name, value=value
        )


def test_validate_output_dtype_and_shape_throws_exception_when_value_contains_too_many_items():
    name = "output2"
    value = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.int32)

    with pytest.raises(
        ValueError,
        match=r"Returned output `output2` for model `Foo` has invalid shapes at one or more positions\. Returned: \(4,\)\. Expected: \(3,\)\.",
    ):
        validate_output_dtype_and_shape(
            model_config=MY_MODEL_CONFIG, model_outputs=MY_MODEL_OUTPUTS, name=name, value=value
        )
