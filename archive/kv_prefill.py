from vllm import LLM, SamplingParams

# mistralai/mistral-7b-v0.1 distilgpt2
llm = LLM(model="mistralai/mistral-7b-v0.1", max_model_len=4096, block_size=64, disable_sliding_window=True, enable_prefix_caching=True)

base_text = "This is a very long document containing a lot of information, discussing various topics in depth. "
repeat_count = 3000 // len(base_text.split())  # 估算需要重复多少次
document = (base_text * repeat_count).strip()
questions = ["Question 2, Are you ok ?", 
                "Question 1, How are you ?", 
            #  "Question 2, Are you ok ?", 
             "Question 3, How old are you ?", 
             "Question 4, Are you not ok ?"]

# 先计算文档的 KV cache
# doc_output = llm.generate(document)
# doc_kv_cache = doc_output.kv_cache  # 获取 KV cache 参考

results = []
for question in questions:
    prompt = document + "\n" + question
    output = llm.generate(prompt)  # 复用 KV cache
    results.append(output)
    metrics = output[0].metrics
    first_token_latency = metrics.first_token_time - metrics.arrival_time  # 计算首 token 生成时间

    print(f"First token latency for '{question}': {first_token_latency:.4f} seconds")

# print(results)

irrelevant_text = "This is a very long document containing a lot of information. "
repeat_count = 3000 // len(irrelevant_text.split())
irrelevant_document = (irrelevant_text * repeat_count).strip()

output = llm.generate(irrelevant_document)
metrics = output[0].metrics
first_token_latency = metrics.first_token_time - metrics.arrival_time
print(f"First token latency for 'irrelevant_text': {first_token_latency:.4f} seconds")
