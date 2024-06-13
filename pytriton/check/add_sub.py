#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""Add_sub example model for checking corectness of triton environment."""

import argparse
import logging
import pathlib
import signal
import sys

import numpy as np

from pytriton.check.utils import ScriptThread
from pytriton.client import ModelClient
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

logger = logging.getLogger("check.add_sub_example")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
add_script_path = [sys.executable, "pytriton/check/add_sub.py"]


@batch
def _add_sub(**inputs):
    a_batch, b_batch = inputs.values()
    add_batch = a_batch + b_batch
    sub_batch = a_batch - b_batch
    return {"add": add_batch, "sub": sub_batch}


def prepare_triton(workspace: pathlib.Path):
    """Prepare triton server with AddSub model."""
    triton = Triton(workspace=str(workspace.resolve()))
    triton.run()
    logger.info("Loading AddSub model")
    triton.bind(
        model_name="AddSub",
        infer_func=_add_sub,
        inputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        outputs=[
            Tensor(name="add", dtype=np.float32, shape=(-1,)),
            Tensor(name="sub", dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128),
        strict=True,
    )
    return triton


def infer_add_sub_model():
    """Infer AddSub model."""
    batch_size = 2
    a_batch = np.ones((batch_size, 1), dtype=np.float32)
    b_batch = np.ones((batch_size, 1), dtype=np.float32)

    logger.info(f"a: {a_batch.tolist()}")
    logger.info(f"b: {b_batch.tolist()}")

    with ModelClient("localhost", "AddSub") as client:
        logger.info("Sending inference request")
        result_batch = client.infer_batch(a_batch, b_batch)

    for output_name, data_batch in result_batch.items():
        logger.info(f"{output_name}: {data_batch.tolist()}")


def serve_triton(workspace: pathlib.Path):
    """Serve triton server with AddSub model."""
    triton = prepare_triton(workspace)
    logger.info("Serving AddSub model")
    triton.serve()


def add_sub_example_thread(workspace: pathlib.Path, logger: logging.Logger):
    """Run example using external script.

    Args:
        workspace: Workspace path that will be created to store testing output (should not exist)
        logger: logger instance
    """
    logger.info("Running example model using external script")

    with ScriptThread(add_script_path + ["--workspace", str(workspace.resolve())], name="server") as server_thread:
        import time

        time.sleep(3)
        infer_add_sub_model()

        if server_thread.process:
            server_thread.process.send_signal(signal.SIGINT)

        server_thread.join()
        logger.error(server_thread.output)
        if server_thread.returncode not in [
            0,
            -2,
        ]:
            logger.error(f"Server failed - return code {server_thread.returncode}")


def add_sub_example(workspace: pathlib.Path, logger: logging.Logger):
    """Run example in the same process.

    Args:
        workspace: Workspace path that will be created to store testing output (should not exist)
        logger: logger instance
    """
    logger.info("Running example model")
    triton = prepare_triton(workspace)
    infer_add_sub_model()
    triton.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", help="Workspace path", type=str)
    parser.add_argument("--infer", default=False, help="Infer AddSub model", action="store_true")
    args = parser.parse_args()

    if args.infer:
        infer_add_sub_model()
    else:
        serve_triton(pathlib.Path(args.workspace))
