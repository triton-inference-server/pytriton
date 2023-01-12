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

import argparse
import functools
import logging
import os
import pathlib
import pickle
import socket
import struct
import tempfile

# pytype: disable=import-error
import jax
import numpy as np
from jax.experimental import PartitionSpec
from jax.experimental.maps import Mesh
from jax.experimental.pjit import pjit
from opt_utils import MODEL_PARALLEL, get_model, get_params_spec, get_tokenizer, greedy_search, shard_params

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

# pytype: enable=import-error


TRITON_MODEL_NAME = "OPT"
MAX_BATCH_SIZE = 256
PORT = 65432

logger = logging.getLogger("examples.huggingface_opt_multinode_jax.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


def run(model, params, tp, server_ip, num_hosts, host_idx):
    params_spec = get_params_spec(model.config.num_hidden_layers, params)
    mesh_devices = jax.devices()[:tp]

    logger.info(f"Selected devices: {mesh_devices}.")
    params = shard_params(model, params, params_spec, mesh_devices)

    infer = pjit(
        functools.partial(greedy_search, model),
        in_axis_resources=(params_spec, PartitionSpec(None, None)),
        out_axis_resources=None,
        static_argnums=2,
    )

    def _server():
        tokenizer = get_tokenizer()

        def wrapper(params, **inputs):
            text, output_len = inputs.values()
            text = np.char.decode(text.astype("bytes"), "utf-8")
            text = text[:, 0]  # squeeze 2nd axis
            input_ids = tokenizer(text.tolist(), return_tensors="np")["input_ids"].astype(np.int64)
            max_len = input_ids.shape[1] + output_len[0].item()
            batch_size = input_ids.shape[0]

            # open a socket to communicate with workers
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((server_ip, PORT))
                s.listen()

                conn_count = 0
                # wait until all the workers receive input data
                while conn_count < num_hosts - 1:
                    conn, _ = s.accept()
                    with conn:
                        data = pickle.dumps({"max_len": max_len, "batch_size": batch_size, "input_ids": input_ids})
                        conn.sendall(struct.pack(">I", len(data)))
                        conn.sendall(data)

                    conn_count += 1

            with Mesh(np.array(mesh_devices), (MODEL_PARALLEL,)):
                outputs = np.array(infer(params, input_ids, max_len))

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            res = [np.array([decoded])]
            return res

        with Triton() as triton:
            logger.info("Loading OPT model.")
            triton.bind(
                model_name=TRITON_MODEL_NAME,
                infer_func=batch(functools.partial(wrapper, params)),
                inputs=[
                    Tensor(name="input", dtype=np.bytes_, shape=(1,)),
                    Tensor(name="output_length", dtype=np.int64, shape=(1,)),
                ],
                outputs=[
                    Tensor(name="output", dtype=np.bytes_, shape=(1,)),
                ],
                config=ModelConfig(max_batch_size=MAX_BATCH_SIZE),
            )
            # Serve model through Triton Inference Server
            logger.info("Serving inference")
            triton.serve()

    def _worker():
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                input_ids, max_len = None, None
                # try to connect with the server until it send input data
                while input_ids is None or max_len is None:
                    try:
                        s.connect((server_ip, PORT))
                        data_size = struct.unpack(">I", s.recv(4))[0]
                        received_payload = b""
                        reamining_payload_size = data_size
                        while reamining_payload_size != 0:
                            received_payload += s.recv(reamining_payload_size)
                            reamining_payload_size = data_size - len(received_payload)
                        data = pickle.loads(received_payload)
                        max_len, batch_size = data["max_len"], data["batch_size"]
                        input_ids = data["input_ids"].reshape((batch_size, -1))
                    except ConnectionRefusedError:
                        pass

            logger.info(input_ids, max_len)
            with Mesh(np.array(mesh_devices), (MODEL_PARALLEL,)):
                infer(params, input_ids, max_len)

    if host_idx == 0:
        _server()
    else:
        _worker()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, help="Name of the HF model")
    parser.add_argument(
        "--server-addr", type=str, default="localhost:1234", help="Server IP and port pair in form of <ip>:<port>"
    )
    parser.add_argument("--num-hosts", type=int, default=1, help="num of hosts")
    parser.add_argument("--host-idx", type=int, default=0, help="index of current host")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size")

    args = parser.parse_args()

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"

    jax.distributed.initialize(args.server_addr, args.num_hosts, args.host_idx)
    logger.info(f"{jax.devices()=}")
    logger.info(f"{jax.local_devices()=}")

    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = pathlib.Path(tempdir)

        model, params = get_model(args.model_name, tempdir)
        run(
            model=model,
            params=params,
            tp=args.tp,
            server_ip=args.server_addr.split(":")[0],
            num_hosts=args.num_hosts,
            host_idx=args.host_idx,
        )


if __name__ == "__main__":
    main()
