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


os.environ["BLOCK_LOG_FILE_PATH"] = "../result/block_log.txt"

file_path = os.environ["BLOCK_LOG_FILE_PATH"]
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"Deleted: {file_path}")
else:
    print(f"File not found: {file_path}")
    
parser = argparse.ArgumentParser(description="Run vLLM with custom CP_ratio.")
parser.add_argument("--cp_ratio", type=float, default=75.0, help="Custom CP_ratio value (default: 20.0)")
parser.add_argument("--cache_size", type=float, default=10000 * 16, help="Cache Size (Tokens)")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for prompts")
parser.add_argument("--cold_start", type=int, default=20000, help="Cold start round number")
parser.add_argument("--model_config", type=str, default='Qwen14B_WikiQA', 
                    choices=list(MODEL_CONFIGS.keys()),
                    help="Model configuration name (default: DeepseekR1_Wild)")
args = parser.parse_args()

# Step 1: Parameters
# model_config = MODEL_CONFIGS['Qwen2.5_Squad']     # TODO Change dataset & model
# model_config = MODEL_CONFIGS['DeepseekR1_QuALITY']
model_config = MODEL_CONFIGS['DeepseekR1_Wild']
model_config = MODEL_CONFIGS[args.model_config]

block_size = 16
# block_size = 128
avg_prompt_len = model_config['avg_prompt_len']       # model related wikiQA 11170.23
CP_ratio = args.cp_ratio
batch_size = args.batch_size
cache_size = args.cache_size
model_weight = model_config['model_weight']         # model related
non_torch_memory = model_config['non_torch_memory']         # model related
torch_activation = model_config['torch_activation']        # model related
kv_per_16_tokens = model_config['kv_per_16_tokens']          # model related
model_name = model_config['model_name']             # model related
max_model_len = model_config['max_model_len']             # model related
gpu_available_memory = 79.18                        # 22.06
memory_overhead = model_weight + non_torch_memory + torch_activation
# gpu_memory_utilization = ( (avg_prompt_len * CP_ratio * kv_per_16_tokens / 16.0) + memory_overhead ) / 22.06
gpu_memory_utilization = ( (cache_size * kv_per_16_tokens / 16.0) + memory_overhead ) / gpu_available_memory  # TODO: Count by input (delete it)
print("gpu_memory_utilization:", gpu_memory_utilization)

# Step 2: Get Data  # TODO Change dataset
# json_path = "/home/shenyang/tests/synthesis/sample/squad_sampled_texts_with_questions.json"
# json_path = "/home/shenyang/tests/synthesis/sample/wikiQA_sampled.json"
json_path = "./WikiQA/wikiQA_distshift_sampled.json"
# json_path = "/home/shenyang/tests/synthesis/SQuAD/sample/squad_sampled_texts_with_questions.json"
# json_path = "/home/shenyang/tests/synthesis/Quality/sample/quality_sampled_texts_with_questions.json"
# json_path = "/home/shenyang/tests/burst/qwen-bailian-usagetraces-anon/sentences.json"         # TODO Wild Workload

with open(json_path, 'r', encoding='utf-8') as json_file:
    prompts = json.load(json_file)
prompts = prompts[:]

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
# sampling_params = SamplingParams(max_tokens=64, min_tokens=48)
# sampling_params = SamplingParams(max_tokens=32, min_tokens=32)
sampling_params = SamplingParams(max_tokens=1)


# Step 3: Test
results = []
latencies = []
cold_start_printed = False
for i in tqdm(range(0, len(prompts), batch_size)):
    if args.cold_start is not None and i > args.cold_start and not cold_start_printed:
        print("[cold start]")
        cold_start_printed = True
    batch_prompts = prompts[i:i+batch_size]
    input_lens = [tokenizer(p, return_tensors="pt")["input_ids"].shape[1] 
              for p in batch_prompts]
    print(f"Average input length: {sum(input_lens) / len(input_lens):.2f} tokens")
    outputs = llm.generate(batch_prompts, sampling_params)
    with open(file_path, "a") as f:
        f.write("\n")
    
    batch_latencies = []
    for output in outputs:
        metrics = output.metrics
        if metrics.first_token_time is not None and metrics.arrival_time is not None:
            first_token_latency = metrics.first_token_time - metrics.arrival_time
            print(f"First token latency: {first_token_latency:.5f} seconds")
            batch_latencies.append(first_token_latency)
        else:
            print("First token latency: N/A (missing metrics)")

        generated_text = output.outputs[0].text if output.outputs else "No output"
        print(f"Generated token length: {len(tokenizer(generated_text)['input_ids'])} tokens")

    results.append(outputs)
    latencies.extend(batch_latencies)

print('First token latency (s):', [round(lat, 5) for lat in latencies])
print('Average of first token latencies:', round(sum(latencies) / len(latencies), 5), 's')
if args.cold_start is not None:
    cold_start_latencies = latencies[args.cold_start:]
    if cold_start_latencies:
        print('Average of first token latencies (after cold start):', round(sum(cold_start_latencies) / len(cold_start_latencies), 5), 's')
