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
from pytriton.server.triton_server import TritonServer
from pytriton.server.triton_server_config import TritonServerConfig
from pytriton.triton import TRITONSERVER_DIST_DIR
from pytriton.utils.distribution import get_libs_path


def test_triton_endpoints():
    config = TritonServerConfig()
    config.update_config({"grpc_port": 8011, "http_port": 8010, "model-repository": "/tmp/repo"})
    triton_server = TritonServer(
        path=TRITONSERVER_DIST_DIR / "bin/tritonserver", libs_path=get_libs_path(), config=config
    )
    assert triton_server.get_endpoint("grpc") == "grpc://127.0.0.1:8011"
    assert triton_server.get_endpoint("http") == "http://127.0.0.1:8010"
    assert triton_server.get_endpoint("metrics") == "http://127.0.0.1:8002"

    config = TritonServerConfig()
    config.update_config({"grpc_address": "192.168.0.1", "model-repository": "/tmp/repo"})
    triton_server = triton_server = TritonServer(
        path=TRITONSERVER_DIST_DIR / "bin/tritonserver", libs_path=get_libs_path(), config=config
    )
    assert triton_server.get_endpoint("grpc") == "grpc://192.168.0.1:8001"
    assert triton_server.get_endpoint("http") == "http://127.0.0.1:8000"
    assert triton_server.get_endpoint("metrics") == "http://127.0.0.1:8002"

    config = TritonServerConfig()
    config.update_config({"http_address": "192.168.0.1", "model-repository": "/tmp/repo"})
    triton_server = triton_server = TritonServer(
        path=TRITONSERVER_DIST_DIR / "bin/tritonserver", libs_path=get_libs_path(), config=config
    )
    assert triton_server.get_endpoint("grpc") == "grpc://127.0.0.1:8001"
    assert triton_server.get_endpoint("http") == "http://192.168.0.1:8000"
    assert triton_server.get_endpoint("metrics") == "http://192.168.0.1:8002"
