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


# following the steps from NeMo prompt learning tutorial notebook
# https://github.com/NVIDIA/NeMo/blob/stable/tutorials/nlp/Multitask_Prompt_and_PTuning.ipynb

set -xe

# while using ncr.io/nvidia/nemo docker image
export NEMO_REPO_DIR=/workspace/nemo

export DATASETS_DIR=${DATASETS_DIR:-$PWD/datasets}
export SENTIMENT_DIR=${DATASETS_DIR}/sentiment
export ASSISTANT_DIR=${DATASETS_DIR}/assistant

export HF_HOME=${HF_HOME:-$PWD/models}
export REPO_ID=${REPO_ID:-nvidia/nemo-megatron-gpt-1.3B}
export MODEL_FILENAME=${MODEL_FILENAME:-nemo_gpt1.3B_fp16.nemo}

LANGUAGE_MODEL_PATH=""  # will be updated by download_model function

download_and_preprocess_data() {
    # download and preprocess data if not already present
    if [ ! -d ${SENTIMENT_DIR} ]; then
        mkdir -p ${SENTIMENT_DIR}
        wget https://huggingface.co/datasets/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip
        unzip FinancialPhraseBank-v1.0.zip -d ${SENTIMENT_DIR}
        rm FinancialPhraseBank-v1.0.zip
        python3 ${NEMO_REPO_DIR}/scripts/dataset_processing/nlp/financial_phrase_bank/prompt_learning_financial_phrase_bank_preprocessing.py \
        --data-dir ${SENTIMENT_DIR}/FinancialPhraseBank-v1.0
        head -4 ${SENTIMENT_DIR}/FinancialPhraseBank-v1.0/financial_phrase_bank_train.jsonl
    fi

    if [ ! -d ${ASSISTANT_DIR} ]; then
        mkdir -p ${ASSISTANT_DIR}
        wget https://github.com/xliuhw/NLU-Evaluation-Data/archive/master.zip
        unzip master.zip -d ${ASSISTANT_DIR}
        rm master.zip
        python3 ${NEMO_REPO_DIR}/scripts/dataset_processing/nlp/intent_and_slot/prompt_learning_assistant_preprocessing.py \
            --source-dir ${ASSISTANT_DIR}/NLU-Evaluation-Data-master \
            --nemo-format-dir ${ASSISTANT_DIR}/nemo-format \
            --output-dir ${ASSISTANT_DIR}
        head -5 ${ASSISTANT_DIR}/assistant_train.jsonl
        echo '\n=====\n#Intents: ' $(wc -l < ${ASSISTANT_DIR}/nemo-format/dict.intents.csv)
        cat ${ASSISTANT_DIR}/nemo-format/dict.intents.csv

        echo '\n=====\n#Slots: ' $(wc -l < ${ASSISTANT_DIR}/nemo-format/dict.slots.csv)
        cat ${ASSISTANT_DIR}/nemo-format/dict.slots.csv
    fi
}

download_model() {
    # ensure model downloaded from HF Hub and get path to model
    cat >> /tmp/ensure_model.py << EOF
import os
from huggingface_hub import hf_hub_download

downloaded_model_path = hf_hub_download(repo_id=os.environ["REPO_ID"], filename=os.environ["MODEL_FILENAME"])
print(downloaded_model_path)
EOF

    LANGUAGE_MODEL_PATH=$(python3 /tmp/ensure_model.py)
    echo ${LANGUAGE_MODEL_PATH}
    rm /tmp/ensure_model.py
}

train_model() {
    python3 ${NEMO_REPO_DIR}/examples/nlp/language_modeling/megatron_gpt_prompt_learning.py \
        name=sentiment_intent_slot_p_tuning \
        model.global_batch_size=64 \
        model.data.train_ds=["${SENTIMENT_DIR}/FinancialPhraseBank-v1.0/financial_phrase_bank_train.jsonl","${ASSISTANT_DIR}/assistant_train.jsonl"] \
        model.data.validation_ds=["${SENTIMENT_DIR}/FinancialPhraseBank-v1.0/financial_phrase_bank_val.jsonl","${ASSISTANT_DIR}/assistant_val.jsonl"] \
            'model.task_templates=[{taskname:sentiment,prompt_template:"<|VIRTUAL_PROMPT_0|> {sentence} sentiment:{label}",total_virtual_tokens:10,virtual_token_splits:[10],truncate_field:None,answer_only_loss:true,answer_field:label},{taskname:intent_and_slot,prompt_template:"<|VIRTUAL_PROMPT_0|> Predict intent and slot <|VIRTUAL_PROMPT_1|> :\n{utterance}{label}",total_virtual_tokens:10,virtual_token_splits:[7,3],truncate_field:None,answer_only_loss:false}]' \
        model.existing_tasks=[] \
        model.new_tasks=[sentiment,intent_and_slot] \
        model.virtual_prompt_style=p-tuning \
        model.language_model_path=${LANGUAGE_MODEL_PATH}
}


# if run in distributed environment
readonly LOCAL_RANK="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
if [ -n "${LOCAL_RANK}" ]; then
    if [ "${LOCAL_RANK}" = "0" ]; then
        download_and_preprocess_data
        download_model
        touch /tmp/local_rank0_finished
    else
        echo ${LOCAL_RANK} waits for the temporary file to appear
        while [ ! -f /tmp/local_rank0_finished ]; do
            sleep 10
        done
    fi
    train_model
    [ "${LOCAL_RANK}" -eq 0 ] && rm /tmp/local_rank0_finished
else
    download_and_preprocess_data
    download_model
    train_model
fi