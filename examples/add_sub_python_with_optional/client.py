#!/usr/bin/env python3
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
"""Client for add_sub_python sample server."""
import logging

import numpy as np

from pytriton.client import ModelClient

LOGGER = logging.getLogger("examples.add_sub_python_with_optional.client")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

batch_size = 2
a_batch = np.ones((batch_size, 1), dtype=np.float32)
b_batch = np.ones((batch_size, 1), dtype=np.float32)


def main():
    LOGGER.info(f"a: {a_batch.tolist()}")
    LOGGER.info(f"b: {b_batch.tolist()}")
    with ModelClient("localhost", "AddSub") as client:
        LOGGER.info("Sending inference request")
        result_batch1 = client.infer_batch(a_batch, b_batch)
        result_batch2 = client.infer_batch(a=a_batch, b=b_batch, w=np.ones((batch_size, 1), dtype=np.float32) * 3)
        result_batch3 = client.infer_batch(a_batch, b_batch, np.ones((batch_size, 1), dtype=np.float32) * 5)
        result_batch4 = client.infer_batch(
            a_batch,
            b_batch,
            np.ones((batch_size, 1), dtype=np.float32) * 5,
            np.ones((batch_size, 1), dtype=np.float32) * 3,
        )
    res = [result_batch1, result_batch2, result_batch3, result_batch4]
    LOGGER.info("Received inference responses")
    for result_batch in res:
        for output_name, data_batch in result_batch.items():
            LOGGER.info(f"{output_name}: {data_batch.tolist()}")
        LOGGER.info("------------------------")


if __name__ == "__main__":
    main()
