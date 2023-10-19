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
from typing import Iterable
from unittest.mock import Mock

import pytest

from pytriton.exceptions import PyTritonInvalidOperationError
from pytriton.models.manager import ModelManager


def _match_length(models: Iterable, length: int) -> bool:
    items = []
    for m in models:
        items.append(m)

    return len(items) == length


def test_add_model_store_models_in_registry_when_models_have_different_names():
    model1 = Mock(model_name="Test1", model_version=1)
    model2 = Mock(model_name="Test2", model_version=1)

    model_manager = ModelManager(triton_url="")
    model_manager.add_model(model1)
    model_manager.add_model(model2)

    assert _match_length(model_manager.models, 2) is True


def test_add_model_store_models_in_registry_when_models_have_different_versions():
    model1 = Mock(model_name="Test1", model_version=1)
    model2 = Mock(model_name="Test1", model_version=2)

    model_manager = ModelManager(triton_url="")
    model_manager.add_model(model1)
    model_manager.add_model(model2)

    assert _match_length(model_manager.models, 2) is True


def test_add_model_raise_error_when_models_have_same_names_and_versions():
    model1 = Mock(model_name="Test", model_version=1)
    model2 = Mock(model_name="Test", model_version=1)

    model_manager = ModelManager(triton_url="")
    model_manager.add_model(model1)

    with pytest.raises(PyTritonInvalidOperationError, match="Cannot add model with the same name twice."):
        model_manager.add_model(model2)


def test_create_models_call_model_generate_and_setup_when_models_added(mocker):
    model1 = Mock(model_name="Test1", model_version=1)
    model2 = Mock(model_name="Test2", model_version=1)
    mocker.patch.object(model1, "is_alive").return_value = False
    mocker.patch.object(model2, "is_alive").return_value = False

    model_manager = ModelManager(triton_url="")
    load_model_method = mocker.patch.object(model_manager, "_load_model")

    model_manager.add_model(model1)
    model_manager.add_model(model2)
    model_manager.load_models()
    assert load_model_method.call_count == 2


def test_clean_call_clean_on_each_model_and_remove_models_from_registry_when_models_added():
    model1 = Mock(model_name="Test1", model_version=1)
    model2 = Mock(model_name="Test2", model_version=1)

    model_manager = ModelManager(triton_url="")

    model_manager.add_model(model1)
    model_manager.add_model(model2)

    assert _match_length(model_manager.models, 2) is True

    model_manager.clean()

    assert model1.clean.called is True
    assert model2.clean.called is True

    assert _match_length(model_manager.models, 0) is True
