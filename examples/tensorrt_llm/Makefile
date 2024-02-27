# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
SCRIPT_DIR                     ?= $(shell pwd)
WORKSPACE_DIR                  ?= $(shell pwd)
TENSORRT_LLM_REPO_DIR          ?= $(WORKSPACE_DIR)/TensorRT-LLM
HF_MODEL_NAME                  ?=
DEPLOYMENT_NAME                ?= $(shell echo $(HF_MODEL_NAME) | sed 's/[^a-zA-Z0-9_-]/_/g')
HF_MODEL_DIR                   ?= $(WORKSPACE_DIR)/models/$(HF_MODEL_NAME)/base
TP                             ?= 1
PP                             ?= 1
MAX_BATCH_SIZE                 ?= 4
MAX_INPUT_LEN                  ?= 3072
MAX_OUTPUT_LEN                 ?= 1024
DTYPE                          ?= float16
INTERIM_DIR                    ?= $(WORKSPACE_DIR)/models/$(HF_MODEL_NAME)/interim/${DEPLOYMENT_NAME}-tp$(TP)-pp$(PP)-$(DTYPE)
ENGINE_DIR                     ?= $(WORKSPACE_DIR)/models/$(HF_MODEL_NAME)/engine/${DEPLOYMENT_NAME}-tp$(TP)-pp$(PP)-$(DTYPE)-bs$(MAX_BATCH_SIZE)-il$(MAX_INPUT_LEN)-ol$(MAX_OUTPUT_LEN)
BENCHMARK_CONCURRENCY_RANGE    ?= 1:16:2

download-hf-model-%:
	@if [ ! -d $(HF_MODEL_DIR) ]; then \
		HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download $(HF_MODEL_NAME) --local-dir $(HF_MODEL_DIR); \
	fi

convert-checkpoint-%: download-hf-model-%
	rm -rf $(INTERIM_DIR)
	$(CONVERTER_CMD)

build-engine-%: convert-checkpoint-%
	rm -rf $(ENGINE_DIR)
	$(BUILD_CMD)

HOST                           ?= 127.0.0.1
HTTP_PORT                      ?= 8000
GRPC_PORT                      ?= 8001
METRICS_PORT                   ?= 8002

serve-%: build-engine-%
	mpirun -n 1 python3 $(SCRIPT_DIR)/server.py --engine-dir $(ENGINE_DIR) --tokenizer $(HF_MODEL_NAME) --model-name $(DEPLOYMENT_NAME) --host $(HOST) --http-port $(HTTP_PORT) --grpc-port $(GRPC_PORT) --metrics-port $(METRICS_PORT)

benchmark-%:
	perf_analyzer \
		-m $(DEPLOYMENT_NAME) -x 1 -u $(HOST):$(GRPC_PORT) -i grpc -a --streaming -b 1 \
		--concurrency-range $(BENCHMARK_CONCURRENCY_RANGE) \
		--measurement-interval 5000 \
		--measurement-mode count_windows \
		--measurement-request-count 20 \
		--input-data input_data.json

# engines catalog
%-llama2-7b-basic: HF_MODEL_NAME 	= meta-llama/Llama-2-7b-hf
%-llama2-7b-basic: DEPLOYMENT_NAME 	= llama2-7b-basic
%-llama2-7b-basic: CONVERTER_CMD	= python3 $(TENSORRT_LLM_REPO_DIR)/examples/llama/convert_checkpoint.py \
										--model_dir $(HF_MODEL_DIR) \
										--output_dir $(INTERIM_DIR) \
										--tp_size $(TP) \
										--pp_size $(PP) \
										--dtype $(DTYPE)
%-llama2-7b-basic: BUILD_CMD		= trtllm-build \
										--checkpoint_dir $(INTERIM_DIR) \
										--output_dir $(ENGINE_DIR) \
										--max_batch_size $(MAX_BATCH_SIZE) \
										--max_input_len $(MAX_INPUT_LEN) \
										--max_output_len $(MAX_OUTPUT_LEN) \
										--gemm_plugin $(DTYPE) \
										--gpt_attention_plugin $(DTYPE) \
										--context_fmha enable \
										--remove_input_padding enable \
										--paged_kv_cache enable

%-llama2-7b-int8: HF_MODEL_NAME 	= meta-llama/Llama-2-7b-hf
%-llama2-7b-int8: DEPLOYMENT_NAME 	= llama2-7b-int8
%-llama2-7b-int8: CONVERTER_CMD		= python3 $(TENSORRT_LLM_REPO_DIR)/examples/llama/convert_checkpoint.py \
										--model_dir $(HF_MODEL_DIR) \
										--output_dir $(INTERIM_DIR) \
										--tp_size $(TP) \
										--pp_size $(PP) \
										--dtype $(DTYPE) \
										--use_weight_only \
										--weight_only_precision int8 \
										--int8_kv_cache
%-llama2-7b-int8: BUILD_CMD			= trtllm-build \
										--checkpoint_dir $(INTERIM_DIR) \
										--output_dir $(ENGINE_DIR) \
										--max_batch_size $(MAX_BATCH_SIZE) \
										--max_input_len $(MAX_INPUT_LEN) \
										--max_output_len $(MAX_OUTPUT_LEN) \
										--gemm_plugin $(DTYPE) \
										--gpt_attention_plugin $(DTYPE) \
										--weight_only_precision int8 \
										--context_fmha enable \
										--remove_input_padding enable \
										--paged_kv_cache enable \
										--multi_block_mode enable