from vllm import LLM, SamplingParams
import statistics

TEST_ROUND=5

# mistralai/mistral-7b-v0.1 distilgpt2
llm = LLM(model="mistralai/mistral-7b-v0.1", max_model_len=4096, block_size=32, disable_sliding_window=True, enable_prefix_caching=True)

base_text = "This is a very long document containing a lot of information, discussing various topics in depth. "
repeat_count = 3000 // len(base_text.split())  # 估算需要重复多少次
document = (base_text * repeat_count).strip()
questions = ["Question 1?" for _ in range(TEST_ROUND + 2)]

results = []
latencies = []
for question in questions:
    prompt = document + "\n" + question
    output = llm.generate(prompt)
    results.append(output)
    metrics = output[0].metrics
    first_token_latency = metrics.first_token_time - metrics.arrival_time  # 计算首 token 生成时间
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
