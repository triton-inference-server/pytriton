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
import logging

import numpy as np

from pytriton.client import ModelClient

TRITON_MODEL_NAME = "OPT"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-addr", type=str, default="localhost", help="server address")
    parser.add_argument("--input", type=str, required=True, help="input text")
    parser.add_argument("--output-length", type=int, required=True, help="output length")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
    logger = logging.getLogger("examples.huggingface_opt_multinode_jax.client")

    np.random.seed(0)

    output_len = np.array([args.output_length], dtype=np.int64)
    inputs = np.array([args.input])
    inputs = np.char.encode(inputs, "utf-8")

    logger.info(f"output_len.shape={output_len.shape}, inputs.shape={inputs.shape}")

    with ModelClient(args.server_addr, model_name=TRITON_MODEL_NAME) as client:
        result_dict = client.infer_sample(inputs, output_len)

    for output_name, output_data in result_dict.items():
        logger.info(f"{output_name}: {[b.decode() for b in output_data.tolist()]}")


if __name__ == "__main__":
    main()
