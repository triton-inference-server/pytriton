# Copyright (c) 2022 - 2023, NVIDIA CORPORATION. All rights reserved.
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

import filelock

# pytype: disable=import-error
import jax
import numpy as np
from jax.experimental.pjit import pjit
from jax.sharding import Mesh, PartitionSpec
from opt_utils import MODEL_PARALLEL, get_model, get_params_spec, get_tokenizer, greedy_search, shard_params

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

# pytype: enable=import-error


TRITON_MODEL_NAME = "OPT"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
LOGGER = logging.getLogger("jax.server")
LOGGER.setLevel(level=logging.INFO)


def run(model, params, number_of_gpus, max_batch_size, server_ip, port, number_of_nodes, rank):
    params_spec = get_params_spec(model.config.num_hidden_layers, params)

    LOGGER.info(f"Available devices: {jax.local_devices()}.")
    mesh_devices = jax.local_devices()[:number_of_gpus]

    LOGGER.info(f"Selected devices: {mesh_devices}.")
    params = shard_params(model, params, params_spec, mesh_devices)

    LOGGER.info("Initialize model")
    infer = pjit(
        functools.partial(greedy_search, model),
        in_axis_resources=(params_spec, PartitionSpec(None, None)),
        out_axis_resources=None,
        static_argnums=2,
    )

    def _server():
        LOGGER.info("Initialize tokenizer.")
        tokenizer = get_tokenizer()

        LOGGER.info("Initialize socket for communication with worker.")
        # open a socket to communicate with workers
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((server_ip, port))
        s.listen()

        def wrapper(params, **inputs):
            text, output_len = inputs.values()
            text = np.char.decode(text.astype("bytes"), "utf-8")
            text = text[:, 0]  # squeeze 2nd axis
            input_ids = tokenizer(text.tolist(), return_tensors="np")["input_ids"].astype(np.int64)
            max_len = input_ids.shape[1] + output_len[0].item()
            batch_size = input_ids.shape[0]

            conn_count = 0
            # wait until all the workers receive input data
            while conn_count < number_of_nodes - 1:
                LOGGER.debug("Broadcast to workers")
                conn, _ = s.accept()
                with conn:
                    data = pickle.dumps({"max_len": max_len, "batch_size": batch_size, "input_ids": input_ids})
                    conn.sendall(struct.pack(">I", len(data)))
                    conn.sendall(data)

                conn_count += 1

            LOGGER.debug("Collecting outputs")
            with Mesh(np.array(mesh_devices), (MODEL_PARALLEL,)):
                outputs = np.array(infer(params, input_ids, max_len))

            LOGGER.debug(f"Result: {outputs}")
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            LOGGER.debug(f"Decoded result: {decoded}")

            res = [np.array([decoded])]
            return res

        with Triton() as triton:
            LOGGER.info("Loading OPT model.")
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
                config=ModelConfig(max_batch_size=max_batch_size),
            )
            # Serve model through Triton Inference Server
            LOGGER.info("Serving inference")
            triton.serve()

    def _worker():
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                input_ids, max_len = None, None
                # try to connect with the server until it send input data
                while input_ids is None or max_len is None:
                    try:
                        s.connect((server_ip, port))
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

            LOGGER.debug(f"{input_ids}, {max_len}")
            with Mesh(np.array(mesh_devices), (MODEL_PARALLEL,)):
                infer(params, input_ids, max_len)

    if rank == 0:
        LOGGER.info(f"Starting server at rank {rank}")
        _server()
    else:
        LOGGER.info(f"Starting worker at rank {rank}")
        _worker()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the HuggingFace model to serve.",
    )
    parser.add_argument(
        "--head-url",
        type=str,
        default="localhost:12345",
        help="Server IP and port pair in form of <ip>:<port> for head node.",
    )
    parser.add_argument(
        "--socket-port",
        type=int,
        default="65432",
        help="Port for socket communication to push array for compute to all workers.",
    )
    parser.add_argument(
        "--number-of-nodes",
        type=int,
        default=1,
        help="Number of nodes.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Rank of current host - 0 mean the head node.",
    )
    parser.add_argument(
        "--number-of-gpus",
        type=int,
        default=1,
        help="Number of gpus used for model.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=256,
        help="The maximal batch size used for model.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Location of cache to avoid download model for multiple nodes.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    LOGGER.setLevel(log_level)

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"

    head_url = args.head_url
    number_of_nodes = args.number_of_nodes
    rank = args.rank

    LOGGER.info(f"Head url: {head_url}")
    LOGGER.info(f"Number of nodes: {number_of_nodes}")
    LOGGER.info(f"Host rank: {rank}")

    jax.distributed.initialize(head_url, number_of_nodes, rank)
    LOGGER.info(f"{jax.devices()=}")
    LOGGER.info(f"{jax.local_devices()=}")

    with tempfile.TemporaryDirectory() as tempdir:
        cache_dir = args.cache_dir
        if not cache_dir:
            cache_dir = tempdir

        cache_dir = pathlib.Path(cache_dir)
        LOGGER.info(f"Cache location: {cache_dir}")

        lock_file = cache_dir / "lock" / "jax_opt.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)

        lock = filelock.FileLock(lock_file.as_posix())
        LOGGER.info(f"Lock in {lock_file}")
        with lock:
            model, params = get_model(args.model_name, cache_dir)

        server_ip, port = args.head_url.split(":")

        run(
            model=model,
            params=params,
            max_batch_size=args.max_batch_size,
            number_of_gpus=args.number_of_gpus,
            server_ip=server_ip,
            port=int(args.socket_port),
            number_of_nodes=number_of_nodes,
            rank=rank,
        )


if __name__ == "__main__":
    main()
