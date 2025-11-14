import dask.dataframe as dd

splits = {'test': 'data/test-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'train': 'data/train-00000-of-00001.parquet'}
# df = dd.read_parquet("hf://datasets/microsoft/wiki_qa/" + splits["test"])
df = dd.concat([
    dd.read_parquet("hf://datasets/microsoft/wiki_qa/" + splits[k])
    # for k in ["train", "validation", "test"]
    for k in ["test"]
])
df = dd.read_parquet("hf://datasets/Cohere/wikipedia-22-12-en-embeddings/data/train-*-of-*.parquet")

# 行数
num_rows = df.shape[0].compute()
print("Total rows:", num_rows)

# # 统计唯一的 document_title 数量
# unique_titles = df['document_title'].nunique().compute()
# print("Unique document_title count:", unique_titles)

# # print(df.head())
# row = df.head(10).iloc[1]   # 取第一行
# for col, value in row.items():
#     print(f"{col}: {value}\n")


