#!/bin/bash
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
#!/bin/bash
set -xe

HEALTH_FILE=/tmp/health

# Define cleanup method
function cleanup()
{
    rm -f ${HEALTH_FILE}
}

# Create health check file
touch ${HEALTH_FILE}

# Clean file on script exit
trap cleanup SIGINT SIGTERM ERR EXIT

# Initial delay to mark POD as health
sleep ${DELAY}

# Initialize head node information
if [ -z ${POD_NAME} ];
then
  RANK=0
  ADDRESS=localhost
else
  POD_ID=${HOSTNAME##*-}
  RANK=$((${POD_ID} % ${NUMBER_OF_NODES}))
  HEAD_RANK=$((${POD_ID} / ${NUMBER_OF_NODES} * ${NUMBER_OF_NODES}))
  ADDRESS=$(dig +short ${POD_NAME}-${HEAD_RANK}.${POD_NAME}.${CLUSTER_NAME})
fi

# Display node info and head address
echo "RANK: ${RANK}"
echo "HEAD ADDRESS: ${ADDRESS}"

# Append cache flags
if [ -n "${PVC_CACHE}" ];
then
echo "Initializing cache in shared volume ${PVC_CACHE}"
export TORCH_HOME=${PVC_CACHE}/torch
export HF_HOME=${PVC_CACHE}/hf
fi

# Use torchrun to initialize distributed computation
torchrun \
    --nproc_per_node=${NUMBER_OF_GPUS} --nnodes=${NUMBER_OF_NODES} --node_rank=${RANK} \
    --max_restarts=0 \
    --master_addr=${ADDRESS} --master_port=${PORT} \
     /opt/app/server.py \
         --gpus ${NUMBER_OF_GPUS} --nodes=${NUMBER_OF_NODES} \
         --model-repo-id ${MODEL_ID}
