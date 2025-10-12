import asyncio
import json
import os
import random
import numpy as np
import argparse
from tqdm import tqdm

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from model_config import MODEL_CONFIGS

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
random.seed(42)
np.random.seed(42)

# ==== 参数和目录设置 ====
parser = argparse.ArgumentParser(description="Run vLLM with continuous batching at 10 req/s.")
parser.add_argument("--cp_ratio", type=float, default=20.0, help="Custom CP_ratio value (default: 20.0)")
args = parser.parse_args()

model_config = MODEL_CONFIGS['Qwen2.5_Squad']
avg_prompt_len = model_config['avg_prompt_len']
CP_ratio = args.cp_ratio
kv_per_16_tokens = model_config['kv_per_16_tokens']
memory_overhead = model_config['model_weight'] + model_config['non_torch_memory'] + model_config['torch_activation']
gpu_memory_utilization = ((avg_prompt_len * CP_ratio * kv_per_16_tokens / 16.0) + memory_overhead) / 22.06

json_path = "/home/shenyang/tests/synthesis/sample/squad_sampled_texts_with_questions.json"
file_path = "/home/shenyang/tests/result/block_log.txt"
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"Deleted: {file_path}")

with open(json_path, 'r', encoding='utf-8') as f:
    prompts = json.load(f)

# ==== 异步请求函数 ====
async def run_request(engine, tokenizer, sampling_params, prompt):
    # 发起生成请求
    request_id = str(hash(prompt))
    arrival_time = asyncio.get_event_loop().time()
    async for output in engine.generate(prompt, sampling_params, request_id):
        if output.finished:
            first_token_time = output.metrics.first_token_time
            latency = first_token_time - arrival_time

            # 写日志
            async with aiofiles.open(file_path, "a") as f:
                await f.write(f"{prompt}\t{latency:.6f}\n")

            # 打印结果
            print(f"Prompt: {prompt}")
            print(f"First token latency: {latency:.5f} s")
            text = output.outputs[0].text if output.outputs else ""
            print(f"Generated text length: {len(tokenizer(text)['input_ids'])} tokens\n")
            return latency

# ==== 主协程 ====
async def main():
    # 初始化 AsyncLLMEngine
    engine_args = AsyncEngineArgs(
        model=model_config['model_name'],
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=model_config['max_model_len'],
        block_size=16,
        disable_sliding_window=True,
        enable_prefix_caching=True
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = engine.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=1)

    tasks = []
    latencies = []

    # 以 10 req/s 的速率调度
    for prompt in prompts:
        tasks.append(asyncio.create_task(run_request(engine, tokenizer, sampling_params, prompt)))
        await asyncio.sleep(0.1)  # 0.1 s 间隔 ⇒ 10 req/s

    # 等待所有任务完成
    results = await asyncio.gather(*tasks)
    latencies = [lat for lat in results if lat is not None]

    # 汇总
    print("First token latencies:", [round(lat, 5) for lat in latencies])
    print("Average latency:", round(sum(latencies) / len(latencies), 5), "s")

if __name__ == "__main__":
    asyncio.run(main())
