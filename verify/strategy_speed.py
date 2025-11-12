from vllm import LLM, SamplingParams
import statistics
import sys
import torch.distributed as dist
import argparse
import os
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

# block_size = int(sys.argv[1])
TEST_ROUND=300
parser = argparse.ArgumentParser(description="Micro Benchmark")
parser.add_argument("--block_size", type=int, default=16, help="Cold start round number")
parser.add_argument("--prompt_length", type=int, default=512, help="Cold start round number")
args = parser.parse_args()
prompt_length = args.prompt_length
block_size = args.block_size

if dist.is_initialized():
    dist.destroy_process_group()
    
sampling_params = SamplingParams(max_tokens=1)

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
llm = LLM(model=model_name, 
          gpu_memory_utilization=0.61,      # 0.657: 134 CUDA blocks
          max_model_len=25000, 
          block_size=block_size, 
          disable_sliding_window=True, 
          enable_prefix_caching=True
        )
tokenizer = llm.get_tokenizer()
vocab = tokenizer.get_vocab()

# prefixes = ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E", "F", "F", "G", "G", "H", "H", "I", "I"]   #classic
base_text_template = " is a very long document containing a lot of information, discussing various topics. "    # 16 Tokens
repeat_count = int(prompt_length / 16)
assert prompt_length % 16 == 0, "prompt_length must be divisible by 16"

# Get words of 1 token
single_token_words = []
for token, token_id in vocab.items():
    text = tokenizer.decode([token_id]).strip()
    if not text:
        continue
    if len(tokenizer(text)["input_ids"]) == 2 \
        and len(tokenizer(((text + base_text_template) * repeat_count).strip()[:-1])["input_ids"]) == 16 * repeat_count\
        and text not in single_token_words:
        single_token_words.append(text)
    if len(single_token_words) >= TEST_ROUND:
        break

prefixes = []
for w in single_token_words:
    prefixes.extend([w, w])
documents = [((prefix + base_text_template) * repeat_count).strip()[:-1] for prefix in prefixes]


results = []
latencies = []
for idx, document in enumerate(documents):
    print('Prompt', idx)
    prompt = document
    # prompt = 'Generate some long passage for me. At least 100 words'
    print(f"vLLM received input length: {tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]} tokens")
    assert tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1] == 16 * repeat_count
    output = llm.generate(prompt, sampling_params)
    with open("/home/shenyang/tests/result/block_log.txt", "a") as f:
        f.write("\n")
    results.append(output)
    metrics = output[0].metrics
    first_token_latency = metrics.first_token_time - metrics.arrival_time  # 计算首 token 生成时间
    latencies.append(first_token_latency)
    print(f"First token latency: {first_token_latency:.5f} seconds")
    
    generated_text = output[0].outputs[0].text if output[0].outputs else "No output"
    print(f"Generated token length: {len(tokenizer(generated_text)["input_ids"])} characters")

print("Latencies:", list(map(lambda x: f"{x:.5f}", latencies)), "seconds")



valid_latency = latencies[2:]
mean_latency = statistics.mean(valid_latency)
variance_latency = statistics.variance(valid_latency)
std_dev_latency = statistics.stdev(valid_latency)

even_indices_latency = valid_latency[::2]
odd_indices_latency = valid_latency[1::2]
mean_even = statistics.mean(even_indices_latency)
variance_even = statistics.stdev(even_indices_latency)
mean_odd = statistics.mean(odd_indices_latency)
variance_odd = statistics.stdev(odd_indices_latency)

print(f"{mean_latency:.6f} seconds - Mean Latency")
print(f"{variance_latency:.10f} - Variance")
print(f"{std_dev_latency:.6f} - Standard Deviation")
print(f"{mean_even:.6f} seconds - Mean Latency (偶数下标,Miss)")
print(f"{variance_even:.10f} - Standard Deviation (偶数下标,Miss)")
print(f"{mean_odd:.6f} seconds - Mean Latency (奇数下标, Hit)")
print(f"{variance_odd:.10f} - Standard Deviation (奇数下标, Hit)")

# 将两个值写入 microbench.txt
with open("/home/shenyang/tests/verify/result/microbench.txt", "a") as f:
    f.write(f"{mean_even:.6f},{mean_odd:.6f}\n")


