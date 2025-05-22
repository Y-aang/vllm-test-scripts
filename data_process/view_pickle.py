import pickle

# 读取 pickle 文件
data_file = "/home/shenyang/data/archive/doc_query_dict_24000.pkl"
# data_file = "/home/shenyang/data/doc_query_dict.pkl"
with open(data_file, "rb") as pfile:
    doc_query_dict = pickle.load(pfile)

# 统计总条数（passages 数量）
num_passages = len(doc_query_dict)

# 统计所有 queries 总数
num_queries = sum(len(queries) for queries in doc_query_dict.values())

# 打印信息
print(f"📌 文档（passages）总数: {num_passages}")
print(f"📌 关联的查询（queries）总数: {num_queries}")

# 预览部分数据
print("\n🔍 示例数据:")
for i, (passage, queries) in enumerate(doc_query_dict.items()):
    print(f"\nPassage {i+1}: {passage[:200]}...")  # 显示前 200 个字符
    print(f"Queries: {queries[:3]}")  # 只显示前 3 个 queries
    if i == 2:  # 仅预览 3 条数据
        break

query_counts = [str(len(queries)) for queries in doc_query_dict.values()]
print(" ".join(query_counts))  # 以空格分隔，单行输出