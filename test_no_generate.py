from vllm import Benchmark

benchmark = Benchmark(
    model="facebook/opt-13b",
    tokenizer="facebook/opt-13b",
    dtype="float16",
    max_model_len=2048,
    num_prompts=100,
    num_batches=10,
    benchmark_type="prefill"  # 只测 prefill
)

results = benchmark.run()
print(results)