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
import gc
import logging
import sys
import threading
import time

import pytest

from pytriton.client.exceptions import PyTritonClientInvalidUrlError
from pytriton.client.utils import _DEFAULT_NETWORK_TIMEOUT_S, create_client_from_url, get_client_lib_from_client

logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger("test_sync_client_utils")


_GRPC_LOCALHOST_URL = "grpc://localhost:8001"
_HTTP_LOCALHOST_URL = "http://localhost:8000"


def test_create_client_raises_urlparse_error_on_invalid_url():
    with pytest.raises(PyTritonClientInvalidUrlError, match="Invalid url"):
        create_client_from_url(["localhost:8001"])  # pytype: disable=wrong-arg-types

    with pytest.raises(PyTritonClientInvalidUrlError, match="Invalid url"):
        create_client_from_url("https://localhost:8000")

    with pytest.raises(PyTritonClientInvalidUrlError, match="Invalid url"):
        create_client_from_url("invalid_scheme://localhost")

    with pytest.raises(PyTritonClientInvalidUrlError, match="Invalid url"):
        create_client_from_url("http://localhost:foo")


def test_create_http_client_set_timeouts():
    # check default http network and connection timeout
    client = create_client_from_url(_HTTP_LOCALHOST_URL)
    connection_pool = client._client_stub._connection_pool
    assert connection_pool.network_timeout == _DEFAULT_NETWORK_TIMEOUT_S
    assert connection_pool.connection_timeout == _DEFAULT_NETWORK_TIMEOUT_S

    client = create_client_from_url(_HTTP_LOCALHOST_URL, network_timeout_s=10.0)
    connection_pool = client._client_stub._connection_pool
    assert connection_pool.network_timeout == 10.0
    assert connection_pool.connection_timeout == 10.0


@pytest.mark.xfail
def test_create_grpc_client_set_timeouts():
    create_client_from_url(_GRPC_LOCALHOST_URL)
    raise NotImplementedError("Not implemented on tritonclient side")


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_del_of_http_client_does_not_raise_error():
    def _del(client):
        del client

    def _create_client_http_and_delete():
        client = create_client_from_url(_HTTP_LOCALHOST_URL)
        client.close()
        threading.Thread(target=_del, args=(client,)).start()

    _create_client_http_and_delete()
    time.sleep(0.1)
    gc.collect()


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_del_of_grpc_client_does_not_raise_error():
    def _del(client):
        del client

    def _create_client_http_and_delete():
        client = create_client_from_url(_GRPC_LOCALHOST_URL)
        client.close()
        threading.Thread(target=_del, args=(client,)).start()

    _create_client_http_and_delete()
    time.sleep(0.1)
    gc.collect()


def test_get_client_lib_from_client():
    client = create_client_from_url(_GRPC_LOCALHOST_URL)
    client_lib = get_client_lib_from_client(client)
    assert client_lib == sys.modules["tritonclient.grpc"]

    client = create_client_from_url(_HTTP_LOCALHOST_URL)
    client_lib = get_client_lib_from_client(client)
    assert client_lib == sys.modules["tritonclient.http"]
