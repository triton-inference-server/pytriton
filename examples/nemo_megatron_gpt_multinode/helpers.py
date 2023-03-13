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
import pathlib
import socket

import filelock
import huggingface_hub  # pytype: disable=import-error
import numpy as np
import torch  # pytype: disable=import-error
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import (  # pytype: disable=import-error
    MegatronGPTModel,
)
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


def download_and_load_model(repo_id, trainer, lock_path, cache_dir):

    lock_path = pathlib.Path(lock_path)
    lock_path.mkdir(parents=True, exist_ok=True)

    loc_file = lock_path / "nemo_megatron_gpt.lock"
    print(f"Lock file {loc_file}")  # noqa
    lock = filelock.FileLock(loc_file.as_posix())

    with lock:
        print(f"Downloading model to {repo_id}")  # noqa
        repo_dir_path = huggingface_hub.snapshot_download(repo_id, cache_dir=cache_dir)
        print(f"Model downloaded to {repo_dir_path}")  # noqa
        model_path = list(pathlib.Path(repo_dir_path).rglob("*.nemo"))[0]

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # although putting model load in filelock section might significantly increase load time
    # especially while loading large models in slurm multi-node scenario
    # but there might be tokenizer files download which is not distributed jobs safe
    print(f"Initializing model from {model_path}")  # noqa
    model = MegatronGPTModel.restore_from(restore_path=model_path.as_posix(), trainer=trainer)

    print("Freezing model")  # noqa
    model.freeze()
    # Have to turn off activations_checkpoint_method for inference
    try:
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
    print(  # noqa
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
