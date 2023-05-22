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

import pytest

from pytriton.utils.dataclasses import kwonly_dataclass


def test_kwonly_dataclasses():
    @kwonly_dataclass
    @dataclasses.dataclass
    class A:
        a: int
        b: int = 0
        c: str = "hello"

    a = A(a=1, b=2, c="world")
    with pytest.raises(TypeError):
        a = A(1, 2, "world")

    assert isinstance(a, A)
    assert dataclasses.is_dataclass(a)
