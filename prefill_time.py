from vllm import LLM, SamplingParams
from transformers import AutoConfig
import statistics
import sys

prompt_length = int(sys.argv[1])
results_file = sys.argv[2]
latencies_file = sys.argv[3]

block_size = 32
TEST_ROUND=10

# mistralai/mistral-7b-v0.1 distilgpt2
llm = LLM(model="mistralai/mistral-7b-v0.1", 
          gpu_memory_utilization=0.9,
          max_model_len=4096, 
          block_size=block_size, 
          disable_sliding_window=True, 
          enable_prefix_caching=True,
        )
tokenizer = llm.get_tokenizer()

base_text = "This is a sentence of 32 tokens, please follow this prompt to generate some sentences. You can generate any thing that you want. Best luck, guy"
repeat_count = prompt_length // 32
document = (base_text * repeat_count).strip().rsplit(" ", 1)[0]
documents = [document for _ in range(TEST_ROUND + 2)]

results = []
latencies = []
for doc in documents:
    prompt = doc
    print(f"vLLM received input length: {tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]} tokens")
    sampling_params_prefill = SamplingParams(max_tokens=0, temperature=0.7)
    output = llm.prefill(prompt, sampling_params_prefill)
    results.append(output)
    metrics = output[0].metrics
    first_token_latency = metrics.prefill_time - metrics.arrival_time
    latencies.append(first_token_latency)

    print(f"First token latency for '{doc[:15]}': {first_token_latency:.5f} seconds")

print("Latencies:", list(map(lambda x: f"{x:.5f}", latencies)), "seconds")


valid_latency = latencies[-TEST_ROUND:]
mean_latency = statistics.mean(valid_latency)
variance_latency = statistics.variance(valid_latency)
std_dev_latency = statistics.stdev(valid_latency)

print(f"{mean_latency:.6f} seconds - Mean Latency")
print(f"{variance_latency:.10f} - Variance")
print(f"{std_dev_latency:.6f} - Standard Deviation")


# 追加实验统计数据到 results.txt（每行一组实验数据，带空格分隔）
with open(results_file, "a") as f:
    f.write(f"prompt_length: {prompt_length},  mean_latency: {mean_latency:.6f} s,  "
            f"std_dev: {std_dev_latency:.6f} s\n")

# 追加实验 latency 数据到 latencies.txt（每 10 个 latency 换行）
with open(latencies_file, "a") as f:
    f.write(f"prompt_length = {prompt_length},  latencies (s):\n  ")
    for i, latency in enumerate(latencies):
        f.write(f"{latency:.5f},  ")
        if (i + 1) % 10 == 0:  # 每 10 个换行
            f.write("\n  ")
    f.write("\n")  # 末尾换行，便于分隔不同 block_size 记录
