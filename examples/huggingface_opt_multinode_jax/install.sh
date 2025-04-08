#!/usr/bin/env bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
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
set -xe

# Use 0.4.14 raises error
pip install --upgrade "jax[cuda12_pip]!=0.4.14" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --upgrade flax omegaconf sacrebleu SentencePiece tokenizers "transformers>=4.26,!=4.51.0"
