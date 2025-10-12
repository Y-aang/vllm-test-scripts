from vllm import LLM

# 初始化 vLLM 模型
llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", max_model_len=4096, block_size=32, disable_sliding_window=True, enable_prefix_caching=True)

# 获取 tokenizer
tokenizer = llm.get_tokenizer()

# 构造一个输入文本
# tmp_256 = "Those is a very long document containing a lot of information,"
base_text = "Those is a very long document containing a lot of information, discussing various topics in depth. "
repeat_count = 227
# repeat_count = 13
document = (base_text * repeat_count).strip()
document = document + "question: who is the president of the state" + "C"
# document = tmp_256 + document + "question: who is the president of the state" + "C"

# 确保 document 正确生成
print(f"Generated document length (characters): {len(document)}")
print(f"Sample document text: {document[:200]}")

# 计算 vLLM 实际接收到的 token 长度
tokenized_input = tokenizer(document, return_tensors="pt")["input_ids"]
input_length = tokenized_input.shape[1]

print(f"vLLM received input length: {input_length} tokens")  # 打印输入 token 数量
