from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time
import pickle
from tqdm import tqdm
import torch.distributed as dist
import statistics

# 在程序最后显式销毁进程组
if dist.is_initialized():
    dist.destroy_process_group()


with open("./doc_query_dict.pkl", 'rb') as pfile:
    doc_query_dict = pickle.load(pfile)

print('len of data: ', len(doc_query_dict))
    
    
model_name = "mistralai/mistral-7b-v0.1"
block_sizes = [16]
block_results = []
query_times_by_doc = []
generated_length = []


gpu_memory_utilization = 0.95

for block_size in block_sizes:
    try:
        # if 'llm' in locals():
        #     del llm
        #     gc.collect()  # 强制垃圾回收
        #     torch.cuda.empty_cache()  # 清理显存
        
        print(f"Testing block size: {block_size}")
        llm = LLM(model=model_name, 
                  block_size=block_size,
                  gpu_memory_utilization=gpu_memory_utilization,
                  max_model_len=4096,
                  dtype="float16",
                  enable_prefix_caching=True,
                  disable_sliding_window=True
                )

        block_times = []  # 用于记录当前 block_size 的时间
        for doc, queries in tqdm(list(doc_query_dict.items())):
            query_times = []
            answer_length = []
            for query_index, query in enumerate(queries):
                prompt = f"Wiki Document: {doc}\nAnswer the following question based on WikiQA:\nQuestion: {query}\nAnswer:"
                # print("The prompt is: ", prompt)
                start_time = time.time()
                # sampling_params = SamplingParams(max_tokens=1024)
                sampling_params = SamplingParams(max_tokens=15)
                output = llm.generate(prompt, sampling_params)
                end_time = time.time()
                
                elapsed_time = end_time - start_time
                generated_text = output[0].outputs[0].text.strip()
                num_generated_tokens = len(generated_text.split())  # 计算生成的 token 数量
                query_times.append(elapsed_time)
                answer_length.append(num_generated_tokens)

                token_time = 0
                if query_index > 0 and num_generated_tokens > 0:
                    token_time = elapsed_time / num_generated_tokens
                    block_times.append(token_time)

                print(f"Block size: {block_size}, Time: {elapsed_time:.2f}s, Tokens: {num_generated_tokens}, Length: {num_generated_tokens}, Time per token: {token_time:.4f}s")
                print(f"Output: {generated_text}\n")
                
            query_times_by_doc.append(query_times)
            generated_length.append(answer_length)
            
        avg_time = sum(block_times) / len(block_times)
        std_dev = statistics.stdev(block_times) if len(block_times) > 1 else 0.0
        print(f"\nAverage time for block size {block_size}: {avg_time:.2f}s")
        print(f"Standard deviation for block size {block_size}: {std_dev:.2f}s")
        block_results.append({"block_size": block_size, "avg_time": avg_time, "std_dev": std_dev})

    except Exception as e:
        print(f"Block size {block_size} failed: {str(e)}")

print("\nSummary of Results:")
print(f"{'Block Size':<15}{'Average Time (s)':<20}{'Standard Deviation (s)':<20}")
for result in block_results:
    print(f"{result['block_size']:<15}{result['avg_time']:<20.2f}{result['std_dev']:<20.2f}")
    
print("\nQuery Times by Doc:")
for doc_index, query_times in enumerate(query_times_by_doc):
    print(f"Doc {doc_index + 1}: {query_times}")
    
print("\nAnswer Length by Doc:")
for doc_index, length in enumerate(generated_length):
    print(f"Doc {doc_index + 1}: {length}")

# import json
# with open("block_size_results.json", "w") as f:
#     json.dump(results, f, indent=2)

# print("\nSummary of Results:")
# for result in results[:5]:
#     print(f"Block size: {result['block_size']}, Time: {result['time']:.2f}s, Query: {result['query']}, Output: {result['output']}")
