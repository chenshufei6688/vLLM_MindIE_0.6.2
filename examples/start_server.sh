#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0
python -m vllm.entrypoints.openai.api_server  \
       --model=/home/data/models/LLaMA3-8B \
       --trust-remote-code \
       --enforce-eager \
       --max-model-len 4096 \
       -tp 1 \
       --port 8006 \
       --block-size 128 