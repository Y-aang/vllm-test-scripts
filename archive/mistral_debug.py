from vllm import LLM, SamplingParams
import pickle

# 读取数据
with open("/home/shenyang/data/doc_query_dict.pkl", 'rb') as pfile:
    doc_query_dict = pickle.load(pfile)

print('Total documents:', len(doc_query_dict))

# 获取第一个文档及其第一个问题
first_doc, first_queries = next(iter(doc_query_dict.items()))
first_query = first_queries[0]

# 构造 prompt
prompt = f"Wiki Document: {first_doc}\nAnswer the following question based on WikiQA:\nQuestion: {first_query}\nAnswer:"
print("\n=== DEBUG: Generated Prompt ===\n")
print(prompt)

# 初始化 vLLM
model_name = "mistralai/mistral-7b-v0.1"
llm = LLM(model=model_name, 
          max_model_len=4096,
          dtype="float16",
          enable_prefix_caching=True,
          disable_sliding_window=True,
        )

# 运行推理
sampling_params = SamplingParams(max_tokens=15)
output = llm.generate(prompt, sampling_params)

# 打印输出
print("\n=== DEBUG: Model Output ===\n")
print(output[0].outputs[0].text.strip())
