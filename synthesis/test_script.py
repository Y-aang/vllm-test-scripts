from vllm import LLM, SamplingParams
import statistics
import sys
import torch.distributed as dist
import json
import random
from tqdm import tqdm
import numpy as np
import os
import argparse
from model_config import MODEL_CONFIGS

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

if dist.is_initialized():
    dist.destroy_process_group()
random.seed(42)
np.random.seed(42)

file_path = "/home/shenyang/tests/result/block_log.txt"
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"Deleted: {file_path}")
else:
    print(f"File not found: {file_path}")
    
parser = argparse.ArgumentParser(description="Run vLLM with custom CP_ratio.")
parser.add_argument("--cp_ratio", type=float, default=75.0, help="Custom CP_ratio value (default: 20.0)")
args = parser.parse_args()

# Step 1: Parameters
model_config = MODEL_CONFIGS['DeepseekR1_QuALITY']
avg_prompt_len = model_config['avg_prompt_len']       # model related wikiQA 11170.23
CP_ratio = args.cp_ratio

block_size = 16
model_weight = model_config['model_weight']         # model related
non_torch_memory = model_config['non_torch_memory']         # model related
torch_activation = model_config['torch_activation']        # model related
kv_per_16_tokens = model_config['kv_per_16_tokens']          # model related
model_name = model_config['model_name']             # model related
max_model_len = model_config['max_model_len']             # model related
memory_overhead = model_weight + non_torch_memory + torch_activation
gpu_memory_utilization = ( (avg_prompt_len * CP_ratio * kv_per_16_tokens / 16.0) + memory_overhead ) / 22.06

# Step 2: Get Data
# json_path = "/home/shenyang/tests/synthesis/sample/squad_sampled_texts_with_questions.json"
# json_path = "/home/shenyang/tests/synthesis/sample/wikiQA_sampled.json"
json_path = "/home/shenyang/tests/synthesis/sample/quality_sampled_texts_with_questions.json"

with open(json_path, 'r', encoding='utf-8') as json_file:
    prompts = json.load(json_file)

#Step 3: initialize LLM
# model_name = "mistralai/mistral-7b-v0.1"
# model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
# model_name = model_config['model_name']
llm = LLM(model=model_name, 
          gpu_memory_utilization=gpu_memory_utilization,
          max_model_len=max_model_len, 
          block_size=block_size, 
          disable_sliding_window=True, 
          enable_prefix_caching=True)
tokenizer = llm.get_tokenizer()
sampling_params = SamplingParams(max_tokens=1)


# Step 3: Test
results = []
latencies = []
for prompt in tqdm(prompts):
    print(f"vLLM received input length: {tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]} tokens")
    output = llm.generate(prompt, sampling_params)
    with open("/home/shenyang/tests/result/block_log.txt", "a") as f:
        f.write("\n")
    results.append(output)
    metrics = output[0].metrics
    first_token_latency = metrics.first_token_time - metrics.arrival_time
    print(f"First token latency: {first_token_latency:.5f} seconds")
    latencies.append(first_token_latency)
    
    generated_text = output[0].outputs[0].text if output[0].outputs else "No output"
    print(f"Generated token length: {len(tokenizer(generated_text)["input_ids"])} characters")

print('First token latency (s):', [round(lat, 5) for lat in latencies])
print('Average of first token latencies:', round(sum(latencies) / len(latencies), 5), 's')