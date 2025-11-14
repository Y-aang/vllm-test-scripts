import pickle
from vllm import LLM, SamplingParams
from tqdm import tqdm

# 初始化 LLM 和 tokenizer（参考 download_wikiqa.py 中的注释写法）
llm = LLM(model="meta-llama/Llama-3.2-3B", 
          gpu_memory_utilization=0.9,
          max_model_len=2000, 
          block_size=16, 
          disable_sliding_window=True, 
          enable_prefix_caching=True
        )
tokenizer = llm.get_tokenizer()

# 读取 pickle 文件（参考 view_pickle.py 的写法）
data_file = "wikiqa_doc_query_dict.pkl"
# data_file = "/home/shenyang/data/doc_query_dict.pkl"
with open(data_file, "rb") as pfile:
    doc_query_dict = pickle.load(pfile)

# 遍历所有 key（passages），计算每个 passage 的 token 长度
token_lengths = []
passage_token_pairs = []  # 存储 (passage, token_length) 对
for passage in tqdm(doc_query_dict.keys()):
    # 参考 download_wikiqa.py 中注释的写法计算 token 长度
    num_tokens = tokenizer(passage, return_tensors="pt")["input_ids"].shape[1]
    token_lengths.append(num_tokens)
    passage_token_pairs.append((passage, num_tokens))

# 计算平均值
if token_lengths:
    avg_length = sum(token_lengths) / len(token_lengths)
    print(f"📌 总 passage 数量: {len(token_lengths)}")
    print(f"📌 Token 长度平均值: {avg_length:.2f}")
    print(f"📌 Token 长度最小值: {min(token_lengths)}")
    print(f"📌 Token 长度最大值: {max(token_lengths)}")
    
    # 找到最短的 wiki
    min_token_length = min(token_lengths)
    shortest_passage = min(passage_token_pairs, key=lambda x: x[1])[0]
    print(f"\n📌 最短的 wiki (Token 长度: {min_token_length}):")
    print("=" * 80)
    print(shortest_passage)
    print("=" * 80)
else:
    print("⚠️ 没有找到任何 passage 数据")

