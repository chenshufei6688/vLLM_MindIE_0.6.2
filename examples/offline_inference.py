# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
# Part of codes in this file was copied from project [vLLM Team][vllm]

import argparse
from vllm import LLM, SamplingParams
from vllm.logger import init_logger

logger = init_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="facebook/opt-125m")

# input prompts for test
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(max_tokens=512, temperature=0)
args = parser.parse_args()
model_path = args.model_path
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,  # number of NPUs to be used
    max_num_seqs=256,  # max batch number
    enforce_eager=True,  # disable CUDA graph mode
    trust_remote_code=True,  # If the model is a custom model not yet available in the HuggingFace transformers library
    worker_use_ray=True,
)

outputs = llm.generate(prompts, sampling_params)
for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    logger.info(
        f"req_num: {i}\nPrompt: {prompt!r}\nGenerated text: {generated_text!r}"
    )