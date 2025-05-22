import time
import threading

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer
)

model_name = "mistralai/mistral-7b-v0.1"  # 替换为你的模型
base_text = "This is a sentence of 32 tokens, please follow this prompt to generate some sentences. You can generate any thing that you want. Best luck, guy"
repeat_count = 64 // 32
document = (base_text * repeat_count).strip().rsplit(" ", 1)[0]
prompt = document

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 如果支持半精度
    device_map="auto"          # 自动放到GPU上
)

# -------------------------------------------------------------------
# 1. 创建一个 TextIteratorStreamer 实例
#    skip_prompt=True 会使得输出中跳过原始输入部分
#    skip_special_tokens=True 会过滤掉特殊token
# -------------------------------------------------------------------
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# -------------------------------------------------------------------
# 2. 定义一个线程来消费 streamer 的输出。每次 streamer 有新的文本片段，
#    都会在这个线程里被处理。
# -------------------------------------------------------------------
start_time = None
first_token_time = None

def stream_consumer():
    global first_token_time
    for new_text in streamer:
        # 如果是第一次获取到生成内容，记录首token到达时间
        if first_token_time is None:
            first_token_time = time.time()
        print(new_text, end="", flush=True)

# -------------------------------------------------------------------
# 3. 在主线程里先启动消费者线程，然后开始调用 generate()
# -------------------------------------------------------------------
consumer_thread = threading.Thread(target=stream_consumer)
consumer_thread.start()

# 记录调用 generate() 前的时间
start_time = time.time()

# 配置一下不使用KV Cache，仅作演示（会大幅降低推理速度）
model.config.use_cache = False

# 调用 generate()，并把 streamer 传进去
input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
output_ids = model.generate(
    **input_ids,
    streamer=streamer,
    max_new_tokens=50,
    use_cache=False  # 再次显式关闭KV Cache
)

consumer_thread.join()  # 等待生成流结束

# -------------------------------------------------------------------
# 4. 计算 time to first token (TTF)
# -------------------------------------------------------------------
if first_token_time is not None:
    ttf = first_token_time - start_time
    print(f"\nTime to first token: {ttf:.4f} seconds")
else:
    print("No tokens were generated.")
