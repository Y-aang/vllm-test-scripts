import asyncio
import time
from datetime import datetime
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
import statistics
import sys
import os

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
block_size = int(sys.argv[1])
TEST_ROUND = 5

class RateLimiter:
    def __init__(self, rate_hz: float):
        self.interval = 1.0 / rate_hz
        self._lock = asyncio.Lock()
        self._last = None

    async def __aenter__(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            if self._last is not None:
                wait = self.interval - (now - self._last)
                if wait > 0:
                    await asyncio.sleep(wait)
            self._last = asyncio.get_event_loop().time()

    async def __aexit__(self, exc_type, exc, tb):
        pass

async def run_request(engine, sampling_params, request_id, prompt):
    # 打印请求开始的美式时间戳
    start = time.time()
    print(f"[{request_id}] start at: {start}")
    async for output in engine.generate(prompt, sampling_params, request_id):
        if output.finished:
            latency = time.time() - start
            return latency, output.outputs[0].text

async def main():
    # 引擎初始化（同之前）
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        gpu_memory_utilization=0.99,
        max_model_len=25000,
        block_size=block_size,
        disable_sliding_window=True,
        enable_prefix_caching=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    sampling_params = SamplingParams(max_tokens=1)

    # 准备 prompts
    prefixes = ["This", "That", "These", "These", "Those"]
    base = " is a very long document containing a lot of information, discussing various topics in depth. "
    repeat_count = 227  # 227 : 4087 tokens, 128 blocks
    # repeat_count = 666  # 227 : 11989 tokens, 749 blocks
    # repeat_count = 15  # 227 : 4087 t/okens, 128 blocks
    documents = [((p + base) * repeat_count).strip() for p in prefixes]

    limiter = RateLimiter(rate_hz=10)
    tasks = []

    for idx, doc in enumerate(documents):
        async def limited(idx=idx, doc=doc):
            async with limiter:
                rid = f"req-{idx}"
                lat, text = await run_request(engine, sampling_params, rid, doc)
                print(f"[{rid}] latency={lat:.4f}s, output_len={len(text)} tokens")
                return lat
        tasks.append(asyncio.create_task(limited()))

    latencies = await asyncio.gather(*tasks)
    print("Latencies:", [f"{lat:.5f}" for lat in latencies], "seconds")

    # 统计最后 TEST_ROUND 次的延迟
    valid = latencies[-TEST_ROUND:]
    mean = statistics.mean(valid)
    var = statistics.variance(valid)
    std = statistics.stdev(valid)
    print(f"\nMean latency (last {TEST_ROUND}): {mean:.6f}s")
    print(f"Variance: {var:.10f}")
    print(f"Std dev: {std:.6f}s")

if __name__ == "__main__":
    asyncio.run(main())
