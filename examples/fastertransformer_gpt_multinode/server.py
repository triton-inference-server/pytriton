#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
"""Text generation server with FasterTransformer GPT model."""
import argparse
import configparser
import dataclasses
import os
import pathlib
import re
import typing

# pytype: disable=import-error
import torch
import transformers

# imports from FasterTransformer repository
# need to add FasterTransformer repository to PYTHONPATH
from examples.pytorch.gpt.utils.gpt import GptInitModelParameters
from examples.pytorch.gpt.utils.parallel_gpt import ParallelGPT
from server_impl import FasterTransformerTritonServer
from utils import download_and_convert_model, patch_gpt_model_if_needed

# pytype: enable=import-error

if not torch.cuda.is_available():
    raise OSError("GPU is needed for the inference")


@dataclasses.dataclass
class HFModelName:
    name: str
    key: str
    model_filename: typing.Optional[str] = None

    @classmethod
    def from_str(cls, hf_model_name):
        hf_model_name, *model_filename_list = hf_model_name.rsplit(":", maxsplit=1)
        return cls(
            name=hf_model_name,
            model_filename="".join(model_filename_list) or None,
            key=re.sub(r"\W", "_", hf_model_name),
        )


def _load_ft_model(
    model_path: pathlib.Path,
    *,
    lib_path: pathlib.Path,
    tp: int,
    pp: int,
    int8_mode: bool,
    data_type: str,
    sparse: bool,
):
    args = argparse.Namespace(
        model_name="gpt",
        pipeline_para_size=pp,
        int8_mode=int8_mode,
        data_type=data_type,
        sparse=sparse,
    )

    config_path = model_path / "config.ini"
    config_reader = configparser.ConfigParser()
    config_reader.read(config_path)
    init_parameters = GptInitModelParameters.from_args(args, config_reader)

    print("\n=============== GPT params ===============")  # noqa
    for key, value in dataclasses.asdict(init_parameters).items():
        print(f"{key}: {value}")  # noqa
    print(f"lib_path: {lib_path}")  # noqa
    print("========================================")  # noqa

    gpt_params = init_parameters.gpt_init_kwargs()
    gpt = ParallelGPT(**gpt_params, lib_path=lib_path)

    patch_gpt_model_if_needed(gpt, config_reader.getint("gpt", "inter_size"), tp=tp)

    if not gpt.load(ckpt_path=model_path.as_posix()):
        raise RuntimeError(f"Could not load {model_path} checkpoint")

    if init_parameters.sparse:
        gpt.sparse()

    gpt.eval()
    gpt.cuda()

    return gpt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-model-name",
        default="gpt2",
        type=HFModelName.from_str,
        dest="hf_model",
        help=(
            "Model repository id on HuggingFace Hub. It can contain filename to select specific model file "
            "(ex. nvidia/nemo-megatron-gpt-5B:nemo_gpt5B_fp16_tp2.nemo)"
        ),
    )
    parser.add_argument("--output-dir", default="ft_models", help="Path to converted FasterTransformer models")
    parser.add_argument("--tp", default=1, type=int, help="Tensor model parallel size")
    parser.add_argument("--pp", default=1, type=int, help="Pipeline model parallel size")
    parser.add_argument(
        "--lib-path",
        default="${FT_REPO_DIR}/build/lib/libth_parallel_gpt.so",
        help="Path of FasterTransformer PyTorch GPT op library",
    )

    args = parser.parse_args()

    torch.distributed.init_process_group(backend=torch.distributed.Backend.MPI)
    rank = torch.distributed.get_rank()

    if rank == 0:
        ft_repo_path = pathlib.Path(os.path.expandvars(args.output_dir)).resolve()
        expected_model_path = ft_repo_path / args.hf_model.key / f"{args.tp}-gpu"
        ft_model_path = (
            download_and_convert_model(args.hf_model.name, output_dir=expected_model_path.parent, tp=args.tp)
            if not expected_model_path.exists()
            else expected_model_path
        )
        object_list = [ft_model_path]
        torch.distributed.broadcast_object_list(object_list, src=0)
    else:
        object_list = [None]
        torch.distributed.broadcast_object_list(object_list, src=0)
        (ft_model_path,) = object_list

    lib_path = pathlib.Path(os.path.expandvars(args.lib_path)).resolve()
    model = _load_ft_model(
        ft_model_path,
        lib_path=lib_path,
        tp=args.tp,
        pp=args.pp,
        int8_mode=False,
        data_type="fp32",
        sparse=False,
    )

    if rank == 0:
        tokenizer = transformers.AutoTokenizer.from_pretrained(ft_model_path)
        print(f"Initialized {tokenizer}")  # noqa
        server = FasterTransformerTritonServer(model, tokenizer, model_name=args.hf_model.key)
        server.serve(ip="0.0.0.0")
    else:
        while True:
            kwargs = FasterTransformerTritonServer.get_remote_kwargs()
            _ = model(**kwargs)


if __name__ == "__main__":
    main()
