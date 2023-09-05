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
import importlib
import logging
import pathlib
import socket
import typing
import warnings
from typing import Dict, Tuple, Type, Union

import filelock
import huggingface_hub  # pytype: disable=import-error
import numpy as np
import omegaconf  # pytype: disable=import-error
import torch  # pytype: disable=import-error
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import (  # pytype: disable=import-error
    MegatronGPTModel,
)
from nemo.collections.nlp.models.language_modeling.megatron_gpt_prompt_learning_model import (  # pytype: disable=import-error
    MegatronGPTPromptLearningModel,
)
from nemo.collections.nlp.models.language_modeling.megatron_t5_model import (  # pytype: disable=import-error
    MegatronT5Model,
)
from nemo.collections.nlp.models.language_modeling.megatron_t5_prompt_learning_model import (  # pytype: disable=import-error
    MegatronT5PromptLearningModel,
)
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector  # pytype: disable=import-error
from nemo.utils.app_state import AppState  # pytype: disable=import-error

from pytriton.exceptions import PyTritonBadParameterError
from pytriton.model_config import Tensor

LOGGER = logging.getLogger("nemo.helpers")


def cast_output(data, required_dtype):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    elif not isinstance(data, np.ndarray):
        data = np.array(data)

        data_is_str = required_dtype in (object, np.object_, bytes, np.bytes_)
        if data_is_str:
            data = np.char.encode(data, "utf-8")

    if data.ndim < 2:
        data = data[..., np.newaxis]
    return data.astype(required_dtype)


def typedict2tensor(
    typedict_class,
    overwrite_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
    defaults: typing.Optional[typing.Dict[str, typing.Any]] = None,
):
    def _map_type(type_):
        if type_ is int:
            return np.int32
        elif type_ is float:
            return np.float32
        elif type_ is bool:
            return np.bool_
        elif type_ is str:
            return bytes
        else:
            raise PyTritonBadParameterError(f"Unknown type {type_}")

    def _get_tensor_params(type_: Type) -> Dict[str, Union[Tuple[int, ...], type]]:
        """
        Returns a shape and a type of Triton tensor. The shape and the type are inferred from a
        Python typing.

        Args:
            type_: a Python typing which should be a single type or a nested ``List``. If `type_` is a usual
                type, then shape is ``(1,)``. If ``type_`` is a nested ``List``, then ``-1`` is added for each
                ``List``. E.g., ``List[int]`` -> ``(1, -1)``, ``List[List[int]]`` -> ``(1, -1, -1)``. Additional
                Please note that all shapes have additional ``(1,)`` leading dimension.

        Returns:
            a dictionary with 2 elements: ``"shape"`` and ``"type"``. ``"type"`` is a numpy type which corresponds
            to ``type_``.
        """
        count = 0
        while typing.get_origin(type_) is list:
            type_ = typing.get_args(type_)[0]
            count += 1
        shape = (1,) + (-1,) * count
        return {"shape": shape, "dtype": _map_type(type_)}

    overwrite_kwargs = overwrite_kwargs or {}
    return tuple(
        Tensor(name=name, **_get_tensor_params(type_), **overwrite_kwargs)
        for name, type_ in typing.get_type_hints(typedict_class).items()
    )


def download_hf_model(repo_id: str, filename: typing.Optional[str] = None) -> pathlib.Path:
    hf_cache_dir = pathlib.Path(huggingface_hub.constants.HUGGINGFACE_HUB_CACHE)
    lock_dir = hf_cache_dir / huggingface_hub.file_download.repo_folder_name(repo_id=repo_id, repo_type="models")
    filename = filename or _get_first_nemo_filename(repo_id)

    lock_path = lock_dir / f"{filename}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Lock file {lock_path}")
    lock = filelock.FileLock(lock_path)

    with lock:
        LOGGER.info(f"Downloading model from https://huggingface.co/{repo_id} filename={filename}")
        model_path = huggingface_hub.hf_hub_download(repo_id, filename=filename)  # set $HF_HOME to set cache dir
        return pathlib.Path(model_path)


def _get_first_nemo_filename(repo_id: str) -> str:
    client = huggingface_hub.HfApi()
    repo_files = client.list_repo_files(repo_id, revision="main")
    nemo_files = [f for f in repo_files if f.endswith(".nemo")]
    if len(nemo_files) == 0:
        raise ValueError(f"Could not find .nemo file in {repo_id}")
    filename = nemo_files[0]
    if len(nemo_files) > 1:
        warnings.warn(
            f"Found more than one .nemo file in {repo_id}. Will be using {filename}. Use --repo-filename to specify the exact file name to use.",
            stacklevel=1,
        )
    return filename


def _get_worker_name() -> str:
    worker_name = socket.gethostname()
    app_state = AppState()
    strategy_initialized = app_state.world_size is not None
    if strategy_initialized:
        worker_name = (
            f"{worker_name}:local={app_state.local_rank},global={app_state.global_rank},dev={app_state.device_id}"
        )

    return worker_name


def _patch_pretrained_cfg(pretrained_cfg, trainer):
    omegaconf.OmegaConf.set_struct(pretrained_cfg, True)
    with omegaconf.open_dict(pretrained_cfg):
        attributes_to_update = {
            "sequence_parallel": False,
            "activations_checkpoint_granularity": None,
            "activations_checkpoint_method": None,
            "precision": trainer.precision,
        }
        for name, value in attributes_to_update.items():
            if hasattr(pretrained_cfg, name):
                pretrained_cfg[name] = value
    return pretrained_cfg


def _patch_prompt_learning_cfg(
    prompt_learning_cfg: omegaconf.DictConfig, pretrained_cfg: omegaconf.DictConfig, pretrained_model_path: pathlib.Path
) -> omegaconf.DictConfig:
    # NeMo prompt learning models doesn't contain target
    # thus need to run define target based on target of pretrained model
    target = {
        _get_target_from_class(MegatronGPTModel): _get_target_from_class(MegatronGPTPromptLearningModel),
        _get_target_from_class(MegatronT5Model): _get_target_from_class(MegatronT5PromptLearningModel),
    }[pretrained_cfg.target]

    # use nemo archive here - pretrained model will be unpacked 2nd time to temporary dir
    with omegaconf.open_dict(prompt_learning_cfg):
        prompt_learning_cfg.language_model_path = pretrained_model_path.as_posix()
        prompt_learning_cfg.target = target

    return prompt_learning_cfg


def _get_target_from_class(target_class) -> str:
    return f"{target_class.__module__}.{target_class.__name__}"


def load_model(
    model_path: pathlib.Path, trainer, *, prompt_learning_model_path: typing.Optional[pathlib.Path] = None
) -> torch.nn.Module:
    worker_name = _get_worker_name()
    LOGGER.debug(f"Loading {model_path} on {worker_name}")

    save_restore_connector = NLPSaveRestoreConnector()
    if model_path.is_dir():
        save_restore_connector.model_extracted_dir = model_path.as_posix()
    pretrained_cfg = save_restore_connector.restore_from(
        None, model_path.as_posix(), return_config=True, trainer=trainer
    )
    if not hasattr(pretrained_cfg, "target"):
        pretrained_cfg["target"] = _get_target_from_class(MegatronGPTModel)

    if prompt_learning_model_path is not None:
        prompt_learning_model_path = pathlib.Path(prompt_learning_model_path)
        save_restore_connector = NLPSaveRestoreConnector()
        prompt_learning_cfg = save_restore_connector.restore_from(
            None, prompt_learning_model_path.as_posix(), return_config=True, trainer=trainer
        )
        prompt_learning_cfg = _patch_prompt_learning_cfg(prompt_learning_cfg, pretrained_cfg, model_path)
        model_to_load_path = prompt_learning_model_path
        override_config = prompt_learning_cfg
    else:
        pretrained_cfg = _patch_pretrained_cfg(pretrained_cfg, trainer)
        model_to_load_path = model_path
        override_config = pretrained_cfg

    module_name, class_name = override_config.target.rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_name), class_name)

    # monkeypatch _build_tokenizer method to be process-safe
    tokenizer_lock = filelock.FileLock(model_path.parent / f"{model_path.name}.tokenizer.lock")

    def _synced_build_tokenizer(self):
        with tokenizer_lock:
            self._original_build_tokenizer()

    model_class._original_build_tokenizer = model_class._build_tokenizer
    model_class._build_tokenizer = _synced_build_tokenizer

    LOGGER.info("Loading model from %s", model_to_load_path.as_posix())
    LOGGER.debug("Override config: %s", override_config)

    model = model_class.restore_from(
        restore_path=model_to_load_path.as_posix(),
        trainer=trainer,
        override_config_path=override_config,
        save_restore_connector=save_restore_connector,
    )

    model.freeze()
    model.training = False
    try:
        # Have to turn off activations_checkpoint_method for inference
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass
    return model


def setup_distributed_environment(trainer):
    def dummy():
        return

    if trainer.strategy.launcher is not None:
        trainer.strategy.launcher.launch(dummy, trainer=trainer)
    trainer.strategy.setup_environment()

    app_state = AppState()

    hostname = socket.gethostname()
    LOGGER.info(
        f"global={app_state.global_rank}/{app_state.world_size} "
        f"local={app_state.local_rank} @ {hostname}:{app_state.device_id} / "
        f"dp={app_state.data_parallel_rank}/{app_state.data_parallel_size} "
        f"tp={app_state.tensor_model_parallel_rank}/{app_state.tensor_model_parallel_size} "
        f"pp={app_state.pipeline_model_parallel_rank}/{app_state.pipeline_model_parallel_size} "
        "vpp="
        f"{getattr(app_state, 'virtual_pipeline_model_parallel_rank', None)}/"
        f"{getattr(app_state, 'virtual_pipeline_model_parallel_size', None)}"
    )
    return app_state
