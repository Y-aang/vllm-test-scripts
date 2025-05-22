from vllm import LLM, SamplingParams
import statistics
import sys
import torch.distributed as dist
import os
from tqdm import tqdm
import pickle
import random
import numpy as np

block_size = int(sys.argv[1])

if dist.is_initialized():
    dist.destroy_process_group()
    
file_path = "/home/shenyang/tests/result/block_log.txt"
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"Deleted: {file_path}")
else:
    print(f"File not found: {file_path}")

sampling_params = SamplingParams(max_tokens=1) # 8000
# mistralai/mistral-7b-v0.1 distilgpt2
llm = LLM(model="mistralai/mistral-7b-v0.1", 
          gpu_memory_utilization=0.95,
          max_model_len=32768, 
          block_size=block_size, 
          disable_sliding_window=True, 
          enable_prefix_caching=True
        )
tokenizer = llm.get_tokenizer()


data_file = "/home/shenyang/data/archive/doc_query_dict_24000.pkl"
with open(data_file, "rb") as pfile:
    doc_query_dict = pickle.load(pfile)
docs = list(doc_query_dict.keys())

def power_law_sampling(num_elements=142, sequence_length=80, exponent=1.0):
    values = np.arange(1, num_elements + 1)
    probabilities = values ** -exponent
    probabilities /= probabilities.sum()
    sampled_indices = np.random.choice(values - 1, size=sequence_length, p=probabilities)
    print("sampled_indices", sampled_indices)
    return [docs[i] for i in sampled_indices]

def power_law_with_hotspot(data, total_length=30, exponent=1.0, 
                           window_size=30, hotspot_ratio=0.10, hotspot_boost=10):
    num_windows = total_length // window_size
    result = []
    num_elements = len(data)
    values = np.arange(1, num_elements + 1)
    base_prob = values ** -exponent
    base_prob /= base_prob.sum()

    for _ in range(num_windows):
        hotspot_indices = np.random.choice(num_elements, size=max(1, int(hotspot_ratio * window_size)), replace=False)
        prob = base_prob.copy()
        prob[hotspot_indices] *= hotspot_boost
        prob /= prob.sum()
        
        sampled_indices = np.random.choice(num_elements, size=window_size, p=prob)
        # print('hotspot_indices', hotspot_indices)
        # print('sampled_indices', sampled_indices)
        result.extend([data[i] for i in sampled_indices])

    return result


random.seed(42)
np.random.seed(42)
# doc_samples = power_law_sampling()
doc_samples = power_law_with_hotspot(data=docs)
doc_samples_cut = []
for prompt in doc_samples:
    words = prompt.strip().split()
    # cut_len = max(1, len(words) * 2 // 3)
    cut_len = max(1, len(words) )
    short_prompt = ' '.join(words[:cut_len])
    doc_samples_cut.append(short_prompt)
selected_prompts = doc_samples_cut
# Print the length
token_lengths = []
for idx, prompt in enumerate(selected_prompts):
    token_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    token_lengths.append(token_len)  # 记录下来
    print(f"Prompt {idx}: {token_len} tokens")
avg_token_len = sum(token_lengths) / len(token_lengths)
print(f"Average prompt token length: {avg_token_len:.2f} tokens")

results = []
latencies = []
for prompt in tqdm(selected_prompts):
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
print('Sum of first token latencies:', round(sum(latencies), 5), 's')