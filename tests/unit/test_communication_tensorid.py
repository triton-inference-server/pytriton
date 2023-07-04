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
import pytest

from pytriton.proxy.communication import TensorId


def test_tensor_id():
    a = TensorId.from_str("/dev/shm/shm0000:0")
    assert a == TensorId("/dev/shm/shm0000", 0)


def test_tensor_id_frozen():
    a = TensorId.from_str("/dev/shm/shm0000:0")
    with pytest.raises(AttributeError):
        a.memory_offset = 2  # type: ignore


def test_tensor_id_parse_str_raises():
    with pytest.raises(ValueError):
        TensorId.from_str("/dev/shm/shm0000:bar")

    with pytest.raises(ValueError):
        TensorId.from_str("/dev/shm/shm0000")

    with pytest.raises(ValueError):
        TensorId.from_str("/dev/shm/shm0000:111:111")


def test_tensor_id_to_str():
    a = TensorId("/dev/shm/shm0000", 0)
    assert str(a) == "/dev/shm/shm0000:0"
