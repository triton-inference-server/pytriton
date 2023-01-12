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
import dataclasses

import numpy as np
import pytest

from pytriton.model_config import Tensor


def test_tensor_set_correct_dtype_when_np_dtype_passed():
    tensor = Tensor(name="variable", dtype=np.float32().dtype, shape=(2, 1))

    assert tensor.dtype == np.float32().dtype.type


def test_tensor_raise_error_when_mutate_field():
    tensor = Tensor(name="variable", dtype=np.float32, shape=(2, 1))

    with pytest.raises(dataclasses.FrozenInstanceError):
        tensor.dtype = np.int32().dtype
