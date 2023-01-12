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

import logging
import pathlib
from dataclasses import dataclass
from typing import Tuple

# pytype: disable=import-error
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import device_put

# from jax import device_put
from jax.experimental import PartitionSpec
from jax.experimental.maps import Mesh
from jax.experimental.pjit import pjit
from modeling_flax_opt import FlaxOPTForCausalLM
from transformers import AutoConfig, AutoTokenizer, FlaxLogitsProcessorList, FlaxMinLengthLogitsProcessor

try:
    from transformers.generation.flax_utils import GreedyState
except ImportError:
    # as in transformers<=4.24.0
    from transformers.generation_flax_utils import GreedyState

from transformers.models.opt import OPTConfig

# pytype: enable=import-error

MODEL_PARALLEL = "mp"


logger = logging.getLogger("examples.huggingface_opt_multinode_jax.opt_utils")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


@dataclass
class Config:
    n_layers: int
    n_heads: int
    d_model: int


CONFIGS = {
    "125M": Config(12, 12, 768),
    "350M": Config(24, 16, 1024),
    "1.3B": Config(24, 32, 2048),
    "2.7B": Config(32, 32, 2560),
    "5B": Config(24, 32, 128 * 32),
    "6.7B": Config(32, 32, 4096),
    "13B": Config(40, 40, 5120),
    "20B": Config(44, 48, 128 * 48),
    "30B": Config(48, 56, 7168),
    "66B": Config(64, 72, 9216),
    "89B": Config(48, 96, 128 * 96),
    "175B": Config(96, 96, 12288),
    "310B": Config(96, 128, 128 * 128),
    "530B": Config(105, 128, 160 * 128),
}


TP_RULES = {
    ("model", "decoder", "embed_positions", "embedding"): PartitionSpec(None, None),
    ("model", "decoder", "embed_tokens", "embedding"): PartitionSpec(None, None),
    ("model", "decoder", "final_layer_norm", "bias"): PartitionSpec(None),
    ("model", "decoder", "final_layer_norm", "scale"): PartitionSpec(None),
    ("model", "decoder", "layers", "{{{LAYER_NUM}}}", "fc1", "bias"): PartitionSpec(MODEL_PARALLEL),
    ("model", "decoder", "layers", "{{{LAYER_NUM}}}", "fc1", "kernel"): PartitionSpec(None, MODEL_PARALLEL),
    ("model", "decoder", "layers", "{{{LAYER_NUM}}}", "fc2", "bias"): PartitionSpec(None),
    ("model", "decoder", "layers", "{{{LAYER_NUM}}}", "fc2", "kernel"): PartitionSpec(MODEL_PARALLEL, None),
    ("model", "decoder", "layers", "{{{LAYER_NUM}}}", "final_layer_norm", "bias"): PartitionSpec(None),
    ("model", "decoder", "layers", "{{{LAYER_NUM}}}", "final_layer_norm", "scale"): PartitionSpec(None),
    ("model", "decoder", "layers", "{{{LAYER_NUM}}}", "self_attn", "k_proj", "bias"): PartitionSpec(MODEL_PARALLEL),
    ("model", "decoder", "layers", "{{{LAYER_NUM}}}", "self_attn", "k_proj", "kernel"): PartitionSpec(
        None, MODEL_PARALLEL
    ),
    ("model", "decoder", "layers", "{{{LAYER_NUM}}}", "self_attn", "q_proj", "bias"): PartitionSpec(MODEL_PARALLEL),
    ("model", "decoder", "layers", "{{{LAYER_NUM}}}", "self_attn", "q_proj", "kernel"): PartitionSpec(
        None, MODEL_PARALLEL
    ),
    ("model", "decoder", "layers", "{{{LAYER_NUM}}}", "self_attn", "v_proj", "bias"): PartitionSpec(MODEL_PARALLEL),
    ("model", "decoder", "layers", "{{{LAYER_NUM}}}", "self_attn", "v_proj", "kernel"): PartitionSpec(
        None, MODEL_PARALLEL
    ),
    ("model", "decoder", "layers", "{{{LAYER_NUM}}}", "self_attn", "out_proj", "bias"): PartitionSpec(None),
    ("model", "decoder", "layers", "{{{LAYER_NUM}}}", "self_attn", "out_proj", "kernel"): PartitionSpec(
        MODEL_PARALLEL, None
    ),
    ("model", "decoder", "layers", "{{{LAYER_NUM}}}", "self_attn_layer_norm", "bias"): PartitionSpec(None),
    ("model", "decoder", "layers", "{{{LAYER_NUM}}}", "self_attn_layer_norm", "scale"): PartitionSpec(None),
}


def get_params_spec(num_layers, params=None):
    param_specs = {}
    for key, spec in TP_RULES.items():
        if any("{{{LAYER_NUM}}}" in n for n in key):
            for layer_num in range(num_layers):
                param_specs[tuple(n.replace("{{{LAYER_NUM}}}", str(layer_num)) for n in key)] = spec
        else:
            param_specs[key] = spec
    params_spec = freeze(unflatten_dict(param_specs))

    if params is not None:
        params_keys = set(flatten_dict(params).keys())
        params_spec_unfreeze = flatten_dict(unfreeze(params_spec))
        for key in tuple(params_spec_unfreeze.keys()):
            if key not in params_keys:
                del params_spec_unfreeze[key]
        params_spec = freeze(unflatten_dict(params_spec_unfreeze))

    return params_spec


def get_config(name: str):
    if name.split("/")[0] == "random":
        name = name.split("/")[-1]
        config = CONFIGS[name]
    else:
        config = AutoConfig.from_pretrained(name)

    return config


def get_model(name: str, tempdir: pathlib.Path) -> Tuple:
    config = get_config(name)
    if name == "facebook/opt-13b":
        config._remove_final_layer_norm = True
    if name.split("/")[0] == "random":
        name = name.split("/")[-1]
        hf_config = OPTConfig(
            hidden_size=config.d_model,
            num_attention_heads=config.n_heads,
            num_hidden_layers=config.n_layers,
            ffn_dim=4 * config.d_model,
        )
        model, params = FlaxOPTForCausalLM(
            config=hf_config,
            dtype=jnp.float16,
            _do_init=False,
        )
    else:
        checkpoint_path = tempdir / "checkpoints" / name
        model, params = FlaxOPTForCausalLM.from_pretrained(
            name,
            config=config,
            dtype=jnp.float16,
            cache_dir=checkpoint_path.as_posix(),
            _do_init=False,
        )

    return model, params


def get_tokenizer(name: str = "facebook/opt-30b"):
    return AutoTokenizer.from_pretrained(name)


def greedy_search(model, params, input_ids, requested_len):
    logger.info("Compiling greedy search....")
    pad_token_id = model.config.pad_token_id
    eos_token_id = model.config.eos_token_id

    batch_size, cur_len = input_ids.shape
    max_length = requested_len

    logits_processor = FlaxLogitsProcessorList()
    logits_processor.append(FlaxMinLengthLogitsProcessor(model.config.min_length, model.config.eos_token_id))

    eos_token_id = jnp.array(eos_token_id)
    pad_token_id = jnp.array(pad_token_id)
    cur_len = jnp.array(cur_len)

    # per batch-item holding current token in loop.
    sequences = jnp.full((batch_size, max_length), pad_token_id, dtype=jnp.int32)
    sequences = lax.dynamic_update_slice(sequences, input_ids, (0, 0))

    # per batch-item state bit indicating if sentence has finished.
    is_sent_finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

    # For Seq2Seq generation, we only need to use the decoder instead of the whole model in generation loop
    # and pass it the `encoder_outputs`, which are part of the `model_kwargs`.
    # initialize model specific kwargs
    model_kwargs = model.prepare_inputs_for_generation(input_ids, max_length)

    # initialize state
    state = GreedyState(
        cur_len=cur_len,
        sequences=sequences,
        running_token=input_ids,
        is_sent_finished=is_sent_finished,
        model_kwargs=model_kwargs,
    )

    def greedy_search_cond_fn(state):
        """state termination condition fn."""
        has_reached_max_length = state.cur_len == max_length
        all_sequence_finished = jnp.all(state.is_sent_finished)
        finish_generation = jnp.logical_or(has_reached_max_length, all_sequence_finished)
        return ~finish_generation

    def greedy_search_body_fn(state):
        """state update fn."""
        logits, cache = model(
            input_ids=state.running_token,
            params=params,
            past_key_values=state.model_kwargs["past_key_values"],
            attention_mask=state.model_kwargs["attention_mask"],
            position_ids=state.model_kwargs["position_ids"],
            return_dict=False,
        )
        logits = logits[:, -1]

        # apply min_length, ...
        logits = logits_processor(state.sequences, logits, state.cur_len)

        next_token = jnp.argmax(logits, axis=-1)

        next_token = next_token * ~state.is_sent_finished + pad_token_id * state.is_sent_finished
        next_is_sent_finished = state.is_sent_finished | (next_token == eos_token_id)
        next_token = next_token[:, None]

        next_sequences = lax.dynamic_update_slice(state.sequences, next_token, (0, state.cur_len))
        next_model_kwargs = model.update_inputs_for_generation(cache, state.model_kwargs)
        return GreedyState(
            cur_len=state.cur_len + 1,
            sequences=next_sequences,
            running_token=next_token,
            is_sent_finished=next_is_sent_finished,
            model_kwargs=next_model_kwargs,
        )

    if input_ids.shape[1] > 1:
        state = greedy_search_body_fn(state)

    state = lax.while_loop(greedy_search_cond_fn, greedy_search_body_fn, state)

    return state.sequences


def shard_params(model, init_params, params_spec, mesh_devices):
    with Mesh(np.array(mesh_devices), (MODEL_PARALLEL,)):
        if init_params is None:
            params = pjit(
                lambda: model.init_weights(model.key, model.input_shape),
                in_axis_resources=None,
                out_axis_resources=params_spec,
            )()
        else:
            new_params = {}
            init_params = flatten_dict(init_params)
            params_spec = flatten_dict(params_spec)

            for key in init_params.keys():
                logger.debug(key)

                init_param = init_params[key]
                init_param = device_put(init_param, mesh_devices[0])

                new_params[key] = pjit(
                    lambda x: x,
                    in_axis_resources=None,
                    out_axis_resources=params_spec[key],
                )(init_param)
            params = freeze(unflatten_dict(new_params))
            params_spec = freeze(unflatten_dict(params_spec))
        num_params_b = np.sum([v.size for v in flatten_dict(params).values()]) / 10**9
        num_params = f"{num_params_b:.2f}B" if num_params_b > 1 else f"{num_params_b * 1000:.2f}M"
        logger.info(f"Number of params: {num_params}")

    return params
