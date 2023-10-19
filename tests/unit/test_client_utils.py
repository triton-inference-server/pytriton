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

import pytest

from pytriton.client.exceptions import PyTritonClientInvalidUrlError
from pytriton.client.utils import TritonUrl

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger("test_client_utils")


def test_parse_triton_url_correctly():
    assert TritonUrl.from_url("localhost:8000") == TritonUrl("http", "localhost", 8000)
    assert TritonUrl.from_url("localhost") == TritonUrl("http", "localhost", 8000)
    assert TritonUrl.from_url("http://abc:8000") == TritonUrl("http", "abc", 8000)
    assert TritonUrl.from_url("grpc://cde:8008") == TritonUrl("grpc", "cde", 8008)
    assert TritonUrl.from_url("http://efg") == TritonUrl("http", "efg", 8000)
    assert TritonUrl.from_url("grpc://ghi") == TritonUrl("grpc", "ghi", 8001)


def test_parse_triton_url_raise_exception_when_schema_is_not_supported():
    with pytest.raises(PyTritonClientInvalidUrlError):
        TritonUrl.from_url("ftp://localhost:8000")
    with pytest.raises(PyTritonClientInvalidUrlError):
        TritonUrl.from_url("https://localhost")


def test_triton_url_with_schema():
    assert TritonUrl.from_url("localhost").with_scheme == "http://localhost:8000"
    assert TritonUrl.from_url("grpc://some:9090").with_scheme == "grpc://some:9090"


def test_triton_url_without_schema():
    assert TritonUrl.from_url("localhost").without_scheme == "localhost:8000"
    assert TritonUrl.from_url("grpc://some:9090").without_scheme == "some:9090"
