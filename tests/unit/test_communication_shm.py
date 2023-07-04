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

import numpy as np

from pytriton.proxy.communication import ShmManager


def test_shm_manager_append_get(mocker):
    shm_manager_write = ShmManager()
    shm_manager_read = ShmManager()

    a = np.zeros((10, 10), dtype=np.float32)
    b = np.array([b"foo", b"longer_bar"])
    c = np.array([b"foo", b"longer_bar"], dtype=object)
    required_buffer_size = sum(shm_manager_write.calc_serialized_size(tensor) for tensor in [a, b, c])
    shm_manager_write.reset_buffer(required_buffer_size)
    a_id = shm_manager_write.append(a)
    b_id = shm_manager_write.append(b)
    c_id = shm_manager_write.append(c)

    a_retrieved = shm_manager_read.get(a_id)
    b_retrieved = shm_manager_read.get(b_id)
    c_retrieved = shm_manager_read.get(c_id)

    np.testing.assert_equal(a, a_retrieved)
    np.testing.assert_equal(b, b_retrieved)
    np.testing.assert_equal(c, c_retrieved)

    spy_close = mocker.spy(shm_manager_write._shm_buffer, "close")
    spy_unlink = mocker.spy(shm_manager_write._shm_buffer, "unlink")

    shm_manager_write.dispose()
    shm_manager_read.dispose()

    spy_close.assert_called_once()
    spy_unlink.assert_called_once()


def test_expand_shared_memory(mocker):
    shm_manager = ShmManager()

    spy_dispose = mocker.spy(shm_manager, "dispose")

    shm_manager.reset_buffer(100)
    shm_manager.reset_buffer(10)
    spy_dispose.assert_not_called()
    shm_manager.reset_buffer(200)
    spy_dispose.assert_called_once()
