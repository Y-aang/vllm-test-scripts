
import dask.dataframe as dd
import collections
import time
import pickle
from tqdm import tqdm
from vllm import LLM, SamplingParams

# llm = LLM(model="mistralai/mistral-7b-v0.1", 
#           gpu_memory_utilization=0.9,
#           max_model_len=2000, 
#           block_size=16, 
#           disable_sliding_window=True, 
#           enable_prefix_caching=True
#         )
# tokenizer = llm.get_tokenizer()

splits = {'test': 'data/test-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'train': 'data/train-00000-of-00001.parquet'}
# df = dd.read_parquet("hf://datasets/microsoft/wiki_qa/" + splits["test"])
df = dd.concat([
    dd.read_parquet("hf://datasets/microsoft/wiki_qa/" + splits[k])
    for k in ["train", "validation", "test"]
])
df_wiki = dd.read_parquet("hf://datasets/Cohere/wikipedia-22-12-en-embeddings/data/train-*-of-*.parquet")

doc_set = set([])
time_start = time.time()
# max_docs = 16000
title_doc_dict = collections.defaultdict(str)
print('here')
cnt = 0
# for title, text in zip(df_wiki['title'], df_wiki['text']) :
for row in df_wiki.itertuples():
    title = row.title
    text = row.text
    cnt += 1
    print(cnt)
    title_doc_dict[title] += text
    # if len(title_doc_dict) > max_docs:
    #     break

# min_token_len = 0
# max_token_len = 24000 
tuples = []
doc_query_dict = collections.defaultdict(list)
doc_id_dict = {}
cnt = 0
for doc_title, question in zip(df['document_title'], df['question']):
    cnt += 1
    print(cnt)
    if doc_title in title_doc_dict and len(doc_title):
        # num_tokens = len(title_doc_dict[doc_title].split())
        # num_tokens = tokenizer(title_doc_dict[doc_title], return_tensors="pt")["input_ids"].shape[1]
        # if num_tokens < max_token_len and num_tokens > min_token_len:
        #     doc_query_dict[title_doc_dict[doc_title]].append(question)
        doc_query_dict[title_doc_dict[doc_title]].append(question)


#doc_query_dict is a dictionary where the keys are passages and values are a list of queries pertaining to a passage.
#this can be used directly as the input dataset to vllm caching
with open("doc_query_dict.pkl", 'wb') as pfile:
    pickle.dump(doc_query_dict, pfile, protocol=pickle.HIGHEST_PROTOCOL)
