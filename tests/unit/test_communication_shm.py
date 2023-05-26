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

import numpy as np

from pytriton.proxy.communication import MetaRequestResponse, ShmManager
from pytriton.proxy.types import Request


def assert_equal_list_of_dicts(a, b):
    assert len(a) == len(b)
    for a_elem, b_elem in zip(a, b):
        assert a_elem.keys() == b_elem.keys()
        for key in a_elem:
            np.testing.assert_equal(a_elem[key], b_elem[key])


def test_numpy_to_from_shm(mocker):
    # floating-point
    shm_manager = ShmManager()
    shm_manager_read = ShmManager()

    a = [
        Request({"a": np.zeros((10, 10), dtype=np.float32), "b": np.array([b"foo", b"longer_bar"])}, {"aaa": 1}),
        Request(
            {"a": np.array(["foo", "longer_bar"]), "c": np.array([b"foo", b"longer_bar"], dtype=object)}, {"ccc": 3}
        ),
    ]

    wrapped_a = shm_manager.to_shm(a, lambda data, req: MetaRequestResponse(data, req.parameters))
    b = shm_manager_read.from_shm(wrapped_a, shm_manager.memory_name(), lambda data, req: Request(data, req.parameters))

    assert_equal_list_of_dicts(a, b)

    spy_a_close = mocker.spy(shm_manager._shm_buffer, "close")
    spy_a_unlink = mocker.spy(shm_manager._shm_buffer, "unlink")

    shm_manager.dispose()
    spy_a_close.assert_called_once()
    spy_a_unlink.assert_called_once()


def test_expand_shared_memory(mocker):
    a = [Request({"a": np.arange(10, dtype=np.float32)})]
    a_larger = [Request({"a": np.arange(100, dtype=np.float32)})]
    a_smaller = [Request({"a": np.arange(100, dtype=np.int16)})]

    shm_manager = ShmManager()
    shm_manager_read = ShmManager()

    spy_dispose = mocker.spy(shm_manager, "dispose")

    wrapped_a = shm_manager.to_shm(a, lambda data, req: MetaRequestResponse(data, req.parameters))
    b = shm_manager_read.from_shm(wrapped_a, shm_manager.memory_name(), lambda data, req: Request(data, req.parameters))
    assert_equal_list_of_dicts(a, b)
    spy_dispose.assert_not_called()

    wrapped_a = shm_manager.to_shm(a_larger, lambda data, req: MetaRequestResponse(data, req.parameters))
    b = shm_manager_read.from_shm(wrapped_a, shm_manager.memory_name(), lambda data, req: Request(data, req.parameters))
    assert_equal_list_of_dicts(a_larger, b)
    spy_dispose.assert_called_once()

    wrapped_a = shm_manager.to_shm(a_smaller, lambda data, req: MetaRequestResponse(data, req.parameters))
    b = shm_manager_read.from_shm(wrapped_a, shm_manager.memory_name(), lambda data, req: Request(data, req.parameters))
    assert_equal_list_of_dicts(a_smaller, b)
    spy_dispose.assert_called_once()

    shm_manager.dispose()
    shm_manager_read.dispose()
