from vllm import LLM, SamplingParams
import statistics
import sys
import torch.profiler as profiler

block_size = int(sys.argv[1])
results_file = sys.argv[2]
latencies_file = sys.argv[3]
batch_size=int(sys.argv[4])

TEST_ROUND=2
trace_dir = "./torch_prof_1024"

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
sampling_params = SamplingParams(max_tokens=1)

base_text = "This is a very long document containing a lot of information, discussing various topics in depth. "
repeat_count = 227
# repeat_count = 13  # TODO 256
document = (base_text * repeat_count).strip()
questions = ["question: who is the president of the state" for _ in range(TEST_ROUND + 2)]
# tmp_256 = "Those is a very long document containing a lot of information,"  # TODO 256

all_results = []
all_latencies = []
schedule = profiler.schedule(wait=0, warmup=1, active=2, repeat=1)
with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    on_trace_ready=profiler.tensorboard_trace_handler(trace_dir),
) as prof:
    for q in questions:
        batch_prompts = []
        for i in range(batch_size):
            tmp_256 = "Those is a very long document containing a lot of information,"
            suffix = chr(ord('B') + i)
            batch_prompts.append(document + q + f" {suffix}")
            # batch_prompts.append(tmp_256 + document + q + f" {suffix}") # TODO 256

        print(f"Batch size: {len(batch_prompts)}")
        input_lens = [tokenizer(p, return_tensors="pt")["input_ids"].shape[1] for p in batch_prompts]
        print("vLLM received input lengths:", input_lens)

        outputs = llm.generate(batch_prompts, sampling_params)
        prof.step()

        # 当前 batch 的结果和 latency
        batch_results = []
        batch_latencies = []
        for output in outputs:
            batch_results.append(output.outputs[0].text)  # 保存生成的文本
            metrics = output.metrics
            first_token_latency = metrics.first_token_time - metrics.arrival_time
            batch_latencies.append(first_token_latency)

        all_results.append(batch_results)
        all_latencies.append(batch_latencies)

        # print(f"Batch results: {batch_results}")
        print(f"First token latencies for batch: {[f'{l:.5f}' for l in batch_latencies]} seconds")

# 打印所有 batch
# print("\nAll Results (by batch):")
# for i, br in enumerate(all_results):
#     print(f"Batch {i}: {br}")

print("\nAll Latencies (by batch):")
for i, bl in enumerate(all_latencies):
    print(f"Batch {i}: {[f'{x:.5f}' for x in bl]}")

# 展平成一维列表用于统计
flat_latencies = [lat for batch in all_latencies for lat in batch]
valid_latency = flat_latencies[-TEST_ROUND * batch_size:]
mean_latency = statistics.mean(valid_latency)
variance_latency = statistics.variance(valid_latency)
std_dev_latency = statistics.stdev(valid_latency)
# 额外：第一轮 batch 的平均 latency
first_batch_latencies = all_latencies[0]
first_batch_mean = statistics.mean(first_batch_latencies)

print(f"{mean_latency:.6f} seconds - Mean Latency")
print(f"{variance_latency:.10f} - Variance")
print(f"{std_dev_latency:.6f} - Standard Deviation")
print(f"{first_batch_mean:.6f} seconds - Mean Latency (first batch)")

# 写 results_file（只记录统计信息，不保存所有文本）
with open(results_file, "a") as f:
    f.write(f"block_size: {block_size}, batch_size: {batch_size}, mean_latency: {mean_latency:.6f} s,  "
            f"std_dev: {std_dev_latency:.6f} s\n")

# 写 latencies_file（按 batch 输出）
with open(latencies_file, "a") as f:
    f.write(f"block_size = {block_size},  latencies (s):\n")
    for i, batch in enumerate(all_latencies):
        f.write(f"  Batch {i}: " + ", ".join([f"{lat:.5f}" for lat in batch]) + "\n")
    f.write("\n")