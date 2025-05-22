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

# mistralai/mistral-7b-v0.1 distilgpt2
llm = LLM(model="mistralai/mistral-7b-v0.1", 
          gpu_memory_utilization=0.95,
          max_model_len=32768, 
          block_size=block_size, 
          disable_sliding_window=True, 
          enable_prefix_caching=True
        )
tokenizer = llm.get_tokenizer()

# Data 1: 4096 prefix
# prefixes = ["This", "This", "That", "These", "Those"]
# base_text_template = " is a very long document containing a lot of information, discussing various topics in depth. "
# repeat_count = 227 * 3
# documents = [((prefix + base_text_template) * repeat_count).strip() for prefix in prefixes]

# Data 2: fix prefix
# documents = [
#     "This is a very long document containing a lot of information, discussing various topics in depth.",
#     "This is a very long document containing a lot of information, so various topics in depth. This is a very long document containing a lot of information, so various topics in depth.",
#     "This is a very long document containing a lot of information, discussing various topics in depth.",
# ]

# data 3: Doc + Query combination
# data_file = "/home/shenyang/data/archive/doc_query_dict_8000.pkl"
# with open(data_file, "rb") as pfile:
#     doc_query_dict = pickle.load(pfile)

# prompts = [
#     f"{doc} {query}"
#     for doc, queries in doc_query_dict.items()
#     for query in queries
# ]
# selected_prompts = prompts[-50:]
# random.seed(42)
# random.shuffle(selected_prompts)

# data 4: power law distribution

data_file = "/home/shenyang/data/archive/doc_query_dict_24000.pkl"
with open(data_file, "rb") as pfile:
    doc_query_dict = pickle.load(pfile)
docs = list(doc_query_dict.keys())

def power_law_sampling(num_elements=8, sequence_length=20, exponent=1.5):
    values = np.arange(1, num_elements + 1)
    probabilities = values ** -exponent
    probabilities /= probabilities.sum()
    sampled_indices = np.random.choice(values - 1, size=sequence_length, p=probabilities)
    print("sampled_indices", sampled_indices)
    return [docs[i] for i in sampled_indices]
random.seed(42)
np.random.seed(42)
doc_samples = power_law_sampling()
selected_prompts = docs

results = []
# latencies = []
for prompt in tqdm(doc_samples):
    print(f"vLLM received input length: {tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]} tokens")
    output = llm.generate(prompt)
    with open("/home/shenyang/tests/result/block_log.txt", "a") as f:
        f.write("\n")
    results.append(output)
    metrics = output[0].metrics
    first_token_latency = metrics.first_token_time - metrics.arrival_time
    # latencies.append(first_token_latency)

    print(f"First token latency: {first_token_latency:.5f} seconds")

# print("Latencies:", list(map(lambda x: f"{x:.5f}", latencies)), "seconds")


# valid_latency = latencies[-TEST_ROUND:]
# mean_latency = statistics.mean(valid_latency)
# variance_latency = statistics.variance(valid_latency)
# std_dev_latency = statistics.stdev(valid_latency)

# print(f"{mean_latency:.6f} seconds - Mean Latency")
# print(f"{variance_latency:.10f} - Variance")
# print(f"{std_dev_latency:.6f} - Standard Deviation")

