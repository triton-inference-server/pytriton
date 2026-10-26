#!/usr/bin/env python3
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
"""e2e test that tensors of every supported dtype pass through the Python backend unchanged under numpy 2"""

import argparse
import logging

LOGGER = logging.getLogger((__package__ or "main").split(".")[-1])
METADATA = {
    "image_name": "nvcr.io/nvidia/pytorch:{TEST_CONTAINER_VERSION}-py3",
    "platforms": ["amd64", "arm64"],
}


def _random_batch(rng, dtype, shape):
    import numpy as np

    if dtype == np.bool_:
        return rng.integers(0, 2, size=shape).astype(np.bool_)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return rng.integers(info.min, info.max, size=shape, dtype=dtype, endpoint=True)
    if np.issubdtype(dtype, np.floating):
        return rng.standard_normal(size=shape).astype(dtype)
    # bytes, sent as an object array
    values = [f"sample-{value}".encode() for value in rng.integers(0, 1_000_000, size=shape).ravel()]
    return np.array(values, dtype=np.object_).reshape(shape)


def main():
    import numpy as np

    from pytriton.check.utils import DEFAULT_LOG_FORMAT, find_free_port
    from pytriton.client import ModelClient
    from pytriton.decorators import batch
    from pytriton.model_config import ModelConfig, Tensor
    from pytriton.triton import Triton, TritonConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-timeout-s", required=False, default=300, type=float, help="Timeout for server and models initialization"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Size of single inference batch")
    parser.add_argument("--seed", type=int, help="PRNG seed", required=False)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=DEFAULT_LOG_FORMAT)
    logging.captureWarnings(True)
    LOGGER.debug("CLI args: %s", args)

    # Triton 24.10 and earlier build the Python backend stub with pybind11 2.10, which garbles tensors under numpy 2.
    numpy_major = int(np.__version__.split(".")[0])
    if numpy_major < 2:
        raise RuntimeError(f"This test must run with numpy 2, found {np.__version__}")
    LOGGER.info("Running with numpy %s", np.__version__)

    dtypes = [
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.bytes_,
    ]

    @batch
    def _identity(**inputs):
        (input_batch,) = inputs.values()
        return {"output": input_batch}

    triton_config = TritonConfig(grpc_port=find_free_port(), http_port=find_free_port(), metrics_port=find_free_port())
    LOGGER.debug("Using %s", triton_config)

    with Triton(config=triton_config) as triton:
        for dtype in dtypes:
            triton.bind(
                model_name=f"identity_{np.dtype(dtype).name}",
                infer_func=_identity,
                inputs=[Tensor(name="input", dtype=dtype, shape=(-1,))],
                outputs=[Tensor(name="output", dtype=dtype, shape=(-1,))],
                config=ModelConfig(max_batch_size=args.batch_size),
                strict=True,
            )
        triton.run()

        rng = np.random.default_rng(args.seed)
        for protocol in ["http", "grpc"]:
            url = f"{protocol}://localhost:{getattr(triton_config, f'{protocol}_port')}"
            for dtype in dtypes:
                model_name = f"identity_{np.dtype(dtype).name}"
                input_batch = _random_batch(rng, dtype, (args.batch_size, 8))
                with ModelClient(url, model_name, init_timeout_s=args.init_timeout_s) as client:
                    result_batch = client.infer_batch(input=input_batch)
                output_batch = result_batch["output"]
                LOGGER.debug("%s over %s: sent %s, received %s", model_name, protocol, input_batch, output_batch)
                if output_batch.dtype != input_batch.dtype:
                    raise AssertionError(
                        f"{model_name} over {protocol}: sent dtype {input_batch.dtype}, received {output_batch.dtype}"
                    )
                np.testing.assert_array_equal(output_batch, input_batch, err_msg=f"{model_name} over {protocol}")
                LOGGER.info("%s over %s: output matches input", model_name, protocol)


if __name__ == "__main__":
    main()
