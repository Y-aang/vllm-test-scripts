from vllm import LLM, SamplingParams
import statistics
import sys

block_size = int(sys.argv[1])
results_file = sys.argv[2]
latencies_file = sys.argv[3]

TEST_ROUND=4

# mistralai/mistral-7b-v0.1 distilgpt2
model_name = "mistralai/mistral-7b-v0.1"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
llm = LLM(model=model_name, 
          gpu_memory_utilization=0.96,
          max_model_len=4096, 
          block_size=block_size, 
          disable_sliding_window=True, 
          enable_prefix_caching=True)
tokenizer = llm.get_tokenizer()

base_text = "This is a very long document containing a lot of information, discussing various topics in depth. "
repeat_count = 227
document = (base_text * repeat_count).strip()
questions = ["question: who is the president of the state" for _ in range(TEST_ROUND + 2)]

results = []
latencies = []
for question in questions:
    prompt = document + question
    print(f"vLLM received input length: {tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]} tokens")
    output = llm.generate(prompt)
    results.append(output)
    metrics = output[0].metrics
    first_token_latency = metrics.first_token_time - metrics.arrival_time
    latencies.append(first_token_latency)

    print(f"First token latency for '{question}': {first_token_latency:.5f} seconds")

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
    f.write(f"block_size: {block_size},  mean_latency: {mean_latency:.6f} s,  "
            f"std_dev: {std_dev_latency:.6f} s\n")

# 追加实验 latency 数据到 latencies.txt（每 10 个 latency 换行）
with open(latencies_file, "a") as f:
    f.write(f"block_size = {block_size},  latencies (s):\n  ")
    for i, latency in enumerate(latencies):
        f.write(f"{latency:.5f},  ")
        if (i + 1) % 10 == 0:  # 每 10 个换行
            f.write("\n  ")
    f.write("\n")  # 末尾换行，便于分隔不同 block_size 记录
