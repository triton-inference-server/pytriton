#!/usr/bin/env bash
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

set -x

THIS_SCRIPT_DIR="$(realpath --relative-to="${PWD}" "$(dirname "$0")")"
MODEL_UNDER_TEST="${MODEL_UNDER_TEST:-"gpt2"}"
DATASET_PATH="${DATASET_PATH:-"${THIS_SCRIPT_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json"}"

# --swap-space Size of the CPU swap space per GPU (in GiB)
# initially in https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py
# sample command was --swap-space 16
SERVER_SCRIPT_ARGS="--model ${MODEL_UNDER_TEST} --swap-space 4 --disable-log-requests"
CLIENT_SCRIPT_ARGS="--tokenizer ${MODEL_UNDER_TEST} --dataset ${DATASET_PATH} --request-rate 128"

pip install numpy "vllm>=0.2.1"

if [ ! -f "$DATASET_PATH" ]; then
    echo "$DATASET_PATH does not exist, downloading it"
    wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -O "$DATASET_PATH"
fi

function benchmark_vllm() {
    python3 -m vllm.entrypoints.api_server \
        ${SERVER_SCRIPT_ARGS} 2>&1 >${THIS_SCRIPT_DIR}/server_output_vllm.log &
    SERVER_PID=$!
    sleep 5

    python3 ${THIS_SCRIPT_DIR}/benchmark_serving.py \
        --backend vllm \
        ${CLIENT_SCRIPT_ARGS} | tee ${THIS_SCRIPT_DIR}/benchmark_output_vllm.log
    CLIENT_EXIT_CODE=$?

    sort outputs-vllm.jsonl > outputs-vllm.jsonl.sorted
    jq . < outputs-vllm.jsonl.sorted > outputs-vllm.jsonl.sorted.jq
    mv outputs-vllm.jsonl.sorted.jq outputs-vllm.jsonl
    rm outputs-vllm.jsonl.sorted

    kill $SERVER_PID
    wait $SERVER_PID || true

    return ${CLIENT_EXIT_CODE}
}

function benchmark_pytriton() {
    python3 ${THIS_SCRIPT_DIR}/server.py \
        ${SERVER_SCRIPT_ARGS} 2>&1 >${THIS_SCRIPT_DIR}/server_output_pytriton.log &
    SERVER_PID=$!
    sleep 5

    # need to use grpc due to no support of http while using decoupled mode
    python3 ${THIS_SCRIPT_DIR}/benchmark_serving.py \
        --backend triton \
        ${CLIENT_SCRIPT_ARGS} | tee ${THIS_SCRIPT_DIR}/benchmark_output_pytriton.log
    CLIENT_EXIT_CODE=$?

    sort outputs-triton.jsonl > outputs-triton.jsonl.sorted
    jq . < outputs-triton.jsonl.sorted > outputs-triton.jsonl.sorted.jq
    mv outputs-triton.jsonl.sorted.jq outputs-triton.jsonl
    rm outputs-triton.jsonl.sorted

    kill $SERVER_PID
    wait $SERVER_PID || true

    return ${CLIENT_EXIT_CODE}
}

benchmark_vllm
benchmark_pytriton


