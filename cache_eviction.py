from vllm import LLM, SamplingParams
import statistics
import sys
import torch.distributed as dist
import os
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

block_size = int(sys.argv[1])

TEST_ROUND=5

if dist.is_initialized():
    dist.destroy_process_group()
    
sampling_params = SamplingParams(max_tokens=1)

# mistralai/mistral-7b-v0.1 distilgpt2
# model_name = "mistralai/mistral-7b-v0.1"
# model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
llm = LLM(model=model_name, 
          gpu_memory_utilization=0.99,      # 0.657: 134 CUDA blocks
          max_model_len=25000, 
          block_size=block_size, 
          disable_sliding_window=True, 
          enable_prefix_caching=True
        )
# HuggingFaceTB/SmolLM2-360M-Instruct
# llm = LLM(model="HuggingFaceTB/SmolLM2-360M-Instruct", 
#           gpu_memory_utilization=0.061,      # 0.657: 134 CUDA blocks
#           max_model_len=4096, 
#           block_size=block_size, 
#           disable_sliding_window=True, 
#           enable_prefix_caching=True
#         )
tokenizer = llm.get_tokenizer()

prefixes = ["This", "That", "These", "These", "Those"]   #classic
# prefixes = ["This", "That", "This", "These", "This"]
base_text_template = " is a very long document containing a lot of information, discussing various topics in depth. "
repeat_count = 227  # 227 : 4087 tokens, 128 blocks
repeat_count = 666  # 227 : 11989 tokens, 749 blocks
repeat_count = 15  # 227 : 4087 t/okens, 128 blocks
documents = [((prefix + base_text_template) * repeat_count).strip() for prefix in prefixes]

# documents = [
#     "This is a very long document containing a lot of information, discussing various topics in depth.",
#     "This is a very long document containing a lot of information, so various topics in depth. This is a very long document containing a lot of information, so various topics in depth.",
#     "This is a very long document containing a lot of information, discussing various topics in depth.",
# ]


results = []
latencies = []
for idx, document in enumerate(documents):
    print('Prompt', idx)
    prompt = document
    # prompt = 'Generate some long passage for me. At least 100 words'
    print(f"vLLM received input length: {tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]} tokens")
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


valid_latency = latencies[-TEST_ROUND:]
mean_latency = statistics.mean(valid_latency)
variance_latency = statistics.variance(valid_latency)
std_dev_latency = statistics.stdev(valid_latency)

print(f"{mean_latency:.6f} seconds - Mean Latency")
print(f"{variance_latency:.10f} - Variance")
print(f"{std_dev_latency:.6f} - Standard Deviation")

