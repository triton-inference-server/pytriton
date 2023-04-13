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
import configparser
import enum
import logging
import os.path
import pathlib
import subprocess
import typing
from typing import Any, Dict, List

import huggingface_hub  # pytype: disable=import-error
import numpy as np
import torch  # pytype: disable=import-error
import transformers  # pytype: disable=import-error

LOGGER = logging.getLogger(__name__)


class WeightDataType(enum.Enum):
    FP32 = "fp32"
    FP16 = "fp16"


def _patch_ft_config(config_path, model_name, infer_tp):
    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)
    config_parser.set(model_name, "tensor_para_size", str(infer_tp))
    with config_path.open("w") as config_file:
        config_parser.write(config_file)


def _create_gpt_tokenizer_config(ft_model_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(ft_model_path)


def _rewrite_tokenizer_config(hf_model_repo_path, ft_model_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_repo_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(ft_model_path)


def _convert_hf_opt2ft(
    hf_model_repo_path: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    trained_tp: int,
    infer_tp: int,
    weight_data_type: WeightDataType,
) -> pathlib.Path:
    convert_script_path = "${FT_REPO_DIR}/examples/pytorch/gpt/utils/huggingface_opt_convert.py"
    convert_script_path = os.path.expandvars(convert_script_path)
    convert_script_path = pathlib.Path(convert_script_path).resolve()
    cmd = [
        "python",
        convert_script_path.as_posix(),
        "-in_file",
        hf_model_repo_path.as_posix(),
        "-saved_dir",
        output_dir.as_posix(),
        "-trained_gpu_num",
        str(trained_tp),
        "-infer_gpu_num",
        str(infer_tp),
        "-weight_data_type",
        weight_data_type.value,
        "-processes",
        "8",
    ]
    LOGGER.debug(f"Running {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    ft_model_path = output_dir / f"{infer_tp}-gpu"
    config_path = ft_model_path / "config.ini"
    _patch_ft_config(config_path, "gpt", infer_tp)
    _rewrite_tokenizer_config(hf_model_repo_path, ft_model_path)

    return ft_model_path


def _convert_hf_gpt2ft(
    hf_model_repo_path: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    trained_tp: int,
    infer_tp: int,
    weight_data_type: WeightDataType,
) -> pathlib.Path:
    convert_script_path = "${FT_REPO_DIR}/examples/pytorch/gpt/utils/huggingface_gpt_convert.py"
    convert_script_path = os.path.expandvars(convert_script_path)
    convert_script_path = pathlib.Path(convert_script_path).resolve()
    cmd = [
        "python",
        convert_script_path.as_posix(),
        "-in_file",
        hf_model_repo_path.as_posix(),
        "-saved_dir",
        output_dir.as_posix(),
        "-trained_gpu_num",
        str(trained_tp),
        "-infer_gpu_num",
        str(infer_tp),
        "-weight_data_type",
        weight_data_type.value,
        "-processes",
        "8",
    ]
    LOGGER.debug(f"Running {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    ft_model_path = output_dir / f"{infer_tp}-gpu"
    config_path = ft_model_path / "config.ini"
    _patch_ft_config(config_path, "gpt", infer_tp)
    _rewrite_tokenizer_config(hf_model_repo_path, ft_model_path)

    return ft_model_path


def _convert_nemo2ft(
    hf_model_repo_path: pathlib.Path,
    output_dir: pathlib.Path,
    infer_tp: int,
    weight_data_type: WeightDataType = WeightDataType.FP32,
) -> pathlib.Path:

    nemo_models_paths = list(hf_model_repo_path.glob("*.nemo"))
    if not nemo_models_paths:
        raise RuntimeError(f"Could not find .nemo files in {hf_model_repo_path}")

    src_model_path = nemo_models_paths[0]
    if len(nemo_models_paths) > 1:
        LOGGER.info(f"There are multiple .nemo files in given HF repo. Using the first one: {src_model_path}")

    convert_script_path = "${FT_REPO_DIR}/examples/pytorch/gpt/utils/nemo_ckpt_convert.py"
    convert_script_path = os.path.expandvars(convert_script_path)
    convert_script_path = pathlib.Path(convert_script_path).resolve()
    cmd = [
        "python",
        convert_script_path.as_posix(),
        "--in-file",
        src_model_path.as_posix(),
        "--saved-dir",
        output_dir.as_posix(),
        "--infer-gpu-num",
        str(infer_tp),
        "--weight-data-type",
        weight_data_type.value,
        "--processes",
        "8",
        "--verbose",
    ]
    LOGGER.debug(f"Running {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    ft_model_path = output_dir / f"{infer_tp}-gpu"
    config_path = ft_model_path / "config.ini"
    _patch_ft_config(config_path, "gpt", infer_tp)
    _create_gpt_tokenizer_config(ft_model_path)

    return ft_model_path


def download_and_convert_model(
    hf_model_name: str,
    *,
    src_filename: typing.Optional[str] = None,
    output_dir: pathlib.Path,
    tp: int,
) -> pathlib.Path:
    assumed_ft_model_path = output_dir / f"{tp}-gpu"
    if assumed_ft_model_path.exists():
        raise ValueError(
            f"Directory {assumed_ft_model_path} already exists - remove it before running conversion script"
        )

    ft_repo_dir = os.environ.get("FT_REPO_DIR")
    assert ft_repo_dir, "Set FT_REPO_DIR environment variable. Script is using converters from FT repository"

    weight_data_type = WeightDataType.FP32

    if src_filename is not None:
        hf_model_path = huggingface_hub.hf_hub_download(hf_model_name, filename=src_filename)
        try:
            transformers.AutoTokenizer.from_pretrained(hf_model_name)
        except (transformers.utils.EntryNotFoundError, OSError):
            pass
        hf_model_path = pathlib.Path(hf_model_path)
        hf_model_repo_path = hf_model_path.parent
    else:
        hf_model_repo_path = huggingface_hub.snapshot_download(hf_model_name)
        hf_model_repo_path = pathlib.Path(hf_model_repo_path)

    if hf_model_name.startswith("nvidia/nemo-megatron-gpt"):
        ft_model_path = _convert_nemo2ft(hf_model_repo_path, output_dir, infer_tp=tp, weight_data_type=weight_data_type)
    else:
        hf_config = transformers.AutoConfig.from_pretrained(hf_model_name)
        try:
            convert_fn = {
                transformers.OPTConfig: _convert_hf_opt2ft,
                transformers.GPT2Config: _convert_hf_gpt2ft,
            }[type(hf_config)]
        except KeyError:
            raise AttributeError(f"Have no converter for model with {hf_config}")

        ft_model_path = convert_fn(
            hf_model_repo_path, output_dir, trained_tp=1, infer_tp=tp, weight_data_type=weight_data_type
        )

    return ft_model_path


def broadcast_inputs(tensors: Dict[str, torch.Tensor], parameters: List[Any], src: int = 0):
    tensors_specs = [(tensor.shape, tensor.dtype) if tensor is not None else None for name, tensor in tensors.items()]
    objects_list = parameters + tensors_specs
    torch.distributed.broadcast_object_list(objects_list, src=src)
    parameters[: len(parameters)] = objects_list[: len(parameters)]
    tensors_specs[: len(tensors_specs)] = objects_list[len(parameters) :]

    for tensor_name, (expected_shape, expected_dtype) in zip(tensors.keys(), tensors_specs):
        tensor = tensors.get(tensor_name)
        if tensor is None or tensor.shape != expected_shape or tensor.dtype != expected_dtype:
            tensors[tensor_name] = torch.empty(expected_shape, dtype=expected_dtype)
        torch.distributed.broadcast(tensors[tensor_name], src=src)


def patch_gpt_model_if_needed(gpt, inter_size: int, tp: int):
    if gpt.weights.local_inter_size * tp != inter_size:
        LOGGER.warning(
            "Patching GPT init state to be able to load weights. "
            f"ffn_size={inter_size} != "
            f"4*tp*local_hidden_units=4*{tp}*{gpt.weights.local_hidden_units}={gpt.weights.local_inter_size}"
        )
        # WAR of missing parametrization of inter_size/ffn_size
        # patch will be introduced in FT repo in upcoming release (>v5.2)
        expected_ffn_kernel1_shape = (gpt.weights.global_hidden_units, gpt.weights.local_inter_size)
        expected_ffn_bias1_shape = (gpt.weights.local_inter_size,)
        expected_ffn_kernel2_shape = (gpt.weights.local_inter_size, gpt.weights.global_hidden_units)
        ffn_kernel1_idxes = np.arange(gpt.weights.layer_num) + 8 * gpt.weights.layer_num
        ffn_bias1_idxes = ffn_kernel1_idxes + gpt.weights.layer_num
        ffn_kernel2_idxes = ffn_bias1_idxes + gpt.weights.layer_num
        # sanity check of idxes
        w = gpt.weights.w
        assert all(w[idx].shape == expected_ffn_kernel1_shape for idx in ffn_kernel1_idxes)
        assert all(w[idx].shape == expected_ffn_bias1_shape for idx in ffn_bias1_idxes)
        assert all(w[idx].shape == expected_ffn_kernel2_shape for idx in ffn_kernel2_idxes)

        gpt.weights.local_inter_size = inter_size / tp
        for idx in ffn_kernel1_idxes:
            w[idx] = torch.zeros(gpt.weights.global_hidden_units, gpt.weights.local_inter_size, dtype=w[idx].dtype)
        for idx in ffn_bias1_idxes:
            w[idx] = torch.zeros(gpt.weights.local_inter_size, dtype=w[idx].dtype)
        for idx in ffn_kernel2_idxes:
            w[idx] = torch.zeros(gpt.weights.local_inter_size, gpt.weights.global_hidden_units, dtype=w[idx].dtype)
