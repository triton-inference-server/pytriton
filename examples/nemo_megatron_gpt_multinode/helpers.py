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
import os
import pathlib
import socket
import typing
import warnings

import filelock
import huggingface_hub  # pytype: disable=import-error
import numpy as np
import omegaconf  # pytype: disable=import-error
import torch  # pytype: disable=import-error
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector  # pytype: disable=import-error
from nemo.utils.app_state import AppState  # pytype: disable=import-error


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


def download_and_load_model(repo_id: str, trainer, *, filename: typing.Optional[str] = None):

    lock_dir = huggingface_hub.cached_assets_path("NeMo", repo_id)
    filename = filename or _get_first_nemo_filename(repo_id)

    lock_path = pathlib.Path(lock_dir) / f"{filename}.lock"
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Lock file {lock_path}")  # noqa: T201
    lock = filelock.FileLock(lock_path)

    with lock:
        print(f"Downloading model from https://huggingface.co/{repo_id} filename={filename}")  # noqa: T201
        model_path = huggingface_hub.hf_hub_download(repo_id, filename=filename)  # set $HF_HOME to set cache dir

        model_path = pathlib.Path(model_path)
        snapshot_name = model_path.parent.name
        cache_subdir = f"extracted/{snapshot_name}/{model_path.name}"

        save_restore_connector = NLPSaveRestoreConnector()
        save_restore_connector.model_extracted_dir = huggingface_hub.cached_assets_path("NeMo", repo_id, cache_subdir)
        model_config_path = pathlib.Path(save_restore_connector.model_extracted_dir) / "model_config.yaml"
        if not os.path.isdir(save_restore_connector.model_extracted_dir) or not model_config_path.is_file():
            print(  # noqa: T201
                f"Extracting nemo model from {model_path} to {save_restore_connector.model_extracted_dir}"
            )
            save_restore_connector._unpack_nemo_file(model_path, save_restore_connector.model_extracted_dir)

    pretrained_cfg = save_restore_connector.restore_from(None, model_path, return_config=True, trainer=trainer)

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
        attributes_to_set_if_missing = {
            # observing that nemo MegatronGPTModel have no target attribute
            "target": "nemo.collections.nlp.models.language_modeling.megatron_gpt_model.MegatronGPTModel",
        }
        for name, value in attributes_to_set_if_missing.items():
            if not hasattr(pretrained_cfg, name):
                pretrained_cfg[name] = value

    module_name, class_name = pretrained_cfg.target.rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_name), class_name)

    # monkeypatch _build_tokenizer method to be process-safe
    def _synced_build_tokenizer(self):
        with lock:
            self._original_build_tokenizer()

    model_class._original_build_tokenizer = model_class._build_tokenizer
    model_class._build_tokenizer = _synced_build_tokenizer

    model = model_class.restore_from(
        restore_path=model_path,
        trainer=trainer,
        override_config_path=pretrained_cfg,
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


def _get_first_nemo_filename(repo_id):
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


def setup_distributed_environment(trainer):
    def dummy():
        return

    if trainer.strategy.launcher is not None:
        trainer.strategy.launcher.launch(dummy, trainer=trainer)
    trainer.strategy.setup_environment()

    app_state = AppState()

    hostname = socket.gethostname()
    print(  # noqa: T201
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
