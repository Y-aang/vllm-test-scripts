import aiohttp
import asyncio
import json
import time

async def vllm_test_like_call_server():
    url = "http://localhost:8000/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": "你好，介绍一下你自己",
        "n": 1,
        "temperature": 0.0,
        "max_tokens": 1,
        "stream": True
    }

    print("posting request...")
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        start_time = time.perf_counter()
        async with session.post(url, headers=headers, json=data) as resp:
            if resp.status != 200:
                print(f"Error: {resp.status} {resp.reason}")
                print(await resp.text())
                return

            buffer = b""
            json_str = None
            first_chunk_time = 0
            first_chunk_received = False

            async for chunk in resp.content.iter_any():
                buffer += chunk

                if not first_chunk_received:
                    first_chunk_time = time.perf_counter() - start_time
                    first_chunk_received = True

                while b"\n" in buffer:
                    json_str, buffer = buffer.split(b"\n", 1)

            # 最后一次完整 JSON
            if json_str:
                output = json.loads(json_str.decode("utf-8"))
                print("[FINAL OUTPUT]", output)
            else:
                output = None

            total_chunk_time = time.perf_counter() - start_time
            print(f"[INFO] First chunk latency: {first_chunk_time:.4f}s")
            print(f"[INFO] Total request time: {total_chunk_time:.4f}s")

asyncio.run(vllm_test_like_call_server())
