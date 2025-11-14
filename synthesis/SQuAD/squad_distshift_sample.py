import json
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm
import random
from pathlib import Path
import os
# os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

random.seed(42)
np.random.seed(42)

# 初始化 vLLM
# model_name = "mistralai/mistral-7b-v0.1"
# model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
# model_name = "microsoft/Phi-3-mini-4k-instruct"
# model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
llm = LLM(model=model_name, 
          gpu_memory_utilization=0.95,
          max_model_len=1024, 
          block_size=16, 
          disable_sliding_window=True, 
          enable_prefix_caching=True)
tokenizer = llm.get_tokenizer()

# 加载 SQuAD 数据集
dataset = load_dataset("rajpurkar/squad")
valid_data = dataset['validation']

# 分组并选择所有的 text，每个 text 只保存所有的 questions
grouped_data = {}
for item in valid_data:
    text = item['context']
    if text not in grouped_data:
        grouped_data[text] = []
    grouped_data[text].append(item['question'])

# 随机选择 1000 个不同的 context (key)
selected_texts = list(grouped_data.keys())
# selected_texts = selected_texts[:1000]  # 不再随机
selected_texts = selected_texts  # 不再随机

# 函数：计算 text + question 的平均长度
def calculate_average_dq_length(grouped_data, selected_texts, tokenizer):
    dq_lengths = []
    text_lengths = []
    print("\nCalculating average length of doc + question (tokens):")
    for text in tqdm(selected_texts, desc="Processing DQ pairs", unit=" context"):
        text_tokens = tokenizer(text, return_tensors="pt")["input_ids"].shape[1]
        text_lengths.append(text_tokens)
        for question in grouped_data[text]:
            question_tokens = tokenizer(question, return_tensors="pt")["input_ids"].shape[1]
            dq_lengths.append(text_tokens + question_tokens)

    # 计算平均长度
    avg_dq_length = sum(dq_lengths) / len(dq_lengths)
    max_dq_length = max(dq_lengths)
    min_dq_length = min(dq_lengths)
    print(f"\n平均 Text + Question 长度 (token 数): {avg_dq_length:.2f}")
    print(f"最大 Text + Question 长度 (token 数): {max_dq_length}")
    print(f"最小 Text + Question 长度 (token 数): {min_dq_length}")
    print(f"Text 的数量: {len(dq_lengths)}")

    # 统计所有 context 的问题数
    question_counts_list = [len(grouped_data[text]) for text in selected_texts]
    print("\nQuestion counts for each context in selected_texts:")
    print(question_counts_list)
    
    print('text lengths:', text_tokens)

    return avg_dq_length

# 调用函数计算平均长度
calculate_average_dq_length(grouped_data, selected_texts, tokenizer)

# 通过 power_law_with_hotspot 对 text 进行采样
def power_law_with_hotspot(data, total_length=1024 * 4, exponent=1.0, 
                           window_size=256, hotspot_ratio=0.10, hotspot_boost=10):
    num_windows = total_length // window_size
    result = []
    num_elements = len(data)
    values = np.arange(1, num_elements + 1)
    base_prob = values ** -exponent
    base_prob /= base_prob.sum()

    for _ in range(num_windows):
        hotspot_indices = np.random.choice(num_elements, size=max(1, int(hotspot_ratio * window_size)), replace=False)
        prob = base_prob.copy()
        prob[hotspot_indices] *= hotspot_boost
        prob /= prob.sum()
        
        sampled_indices = np.random.choice(num_elements, size=window_size, p=prob)
        result.extend([data[i] for i in sampled_indices])

    return result

# 通过 Distribution Shift 采样
def distribution_shift_sampling(data, total_length=1024*4, window_size=1024, 
                                exponent=1.0, shuffle_each_window=True):
    num_windows = total_length // window_size
    result = []

    for _ in range(num_windows):
        # 每个窗口打乱一次
        if shuffle_each_window:
            np.random.shuffle(data)

        num_elements = len(data)
        values = np.arange(1, num_elements + 1)
        base_prob = values ** -exponent
        base_prob /= base_prob.sum()
        
        # 在当前窗口内采样
        sampled_indices = np.random.choice(len(data), size=window_size, p=base_prob)
        result.extend([data[i] for i in sampled_indices])

    return result


def distribution_shift_sampling_batch(data, total_length=1024*8, window_size=1024, batch_size=32,
                                exponent=1.0, shuffle_each_window=True):
    num_windows = total_length // window_size
    result = []

    for _ in range(num_windows):
        # 每个窗口打乱一次
        if shuffle_each_window:
            np.random.shuffle(data)

        num_elements = len(data)
        values = np.arange(1, num_elements + 1)
        base_prob = values ** -exponent
        base_prob /= base_prob.sum()
        
        sampled = []
        for i in range(0, window_size, batch_size):
            # 每个 batch 内不重复
            batch_indices = np.random.choice(
                len(data), size=min(batch_size, num_elements), 
                replace=False, p=base_prob
            )
            sampled.extend([data[j] for j in batch_indices])
        result.extend(sampled)

    return result


# sampled_texts = power_law_with_hotspot(selected_texts)
sampled_texts = distribution_shift_sampling(selected_texts)
# sampled_texts = distribution_shift_sampling_batch(selected_texts)

# 函数：为每个 sampled_texts 随机追加一个 question
def append_random_question(sampled_texts, grouped_data):
    appended_texts = []
    for text in sampled_texts:
        if text in grouped_data and grouped_data[text]:
            chosen_question = random.choice(grouped_data[text])
            appended_texts.append(text + " " + chosen_question)
        else:
            assert False
            appended_texts.append(text)  # 如果没有问题则保留原文本
    return appended_texts

# 使用新函数为 sampled_texts 每个元素随机追加一个 question
appended_texts = append_random_question(sampled_texts, grouped_data)

# 打印前10个示例
print("\nSampled Texts with Appended Questions (First 10):")
for i, text in enumerate(appended_texts[:1]):
    print(f"{i + 1}: {text}\n")

# 可选：保存结果为 JSON（调试用）
BASE_DIR = Path(__file__).resolve().parent
json_path = BASE_DIR / "squad_sampled_texts_with_questions.json"
# json_path = "/home/shenyang/tests/synthesis/SQuAD/sample/squad_sampled_texts_with_questions.json"
with open(json_path, 'w', encoding='utf-8') as json_file:
    json.dump(appended_texts, json_file, ensure_ascii=False, indent=4)

print(f"\n已保存为: {json_path}, 共生成 {len(appended_texts)} 条样本。")
