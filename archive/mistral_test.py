from vllm import LLM, SamplingParams

# 直接使用模型名称
model_name = "mistralai/Mistral-7B-v0.1"  # 替换为你要使用的模型名称

# 初始化 LLM（vLLM 会自动处理模型加载）
llm = LLM(model=model_name, max_model_len=14096, block_size=256)

# 定义推理参数
sampling_params = SamplingParams(
    temperature=0.7,  # 控制生成的随机性
    max_tokens=50,    # 最大生成长度
    top_p=0.9,        # Nucleus Sampling (Top-p)
)

# 输入提示
prompt = "Explain the importance of renewable energy in simple terms."

# 执行推理
output = llm.generate(prompt, sampling_params=sampling_params)

# 打印结果
print("Input prompt:", prompt)
print("Generated output:", output[0])
