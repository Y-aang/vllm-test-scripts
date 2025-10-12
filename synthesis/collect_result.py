import os
import re
import pandas as pd

# 设置变量
model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
dataset_name = "wild"
sample_strategy = "3200"
cache_strategies = ["arc", "dbl", "lru"]
cache_strategies = ["lru", "dbl"]
cache_strategies = ["lru"]

# 初始化结果表，key 为 Cache_Size（如 5、10...）
results = {}

for cache_strategy in cache_strategies:
    dir_path = f"./test/{model_name}/{dataset_name}/{sample_strategy}/{cache_strategy}"
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"{dir_path} not found")

    for filename in os.listdir(dir_path):
        if not filename.endswith(".txt"):
            continue
        cache_size = int(filename[:-4])
        file_path = os.path.join(dir_path, filename)

        with open(file_path, "r") as f:
            lines = f.readlines()

        # 找出最后一次出现的 gpu_hit_rate 和 latency
        hit_rate = None
        latency = None
        for line in reversed(lines):
            if "gpu_hit_rate" in line:
                match = re.search(r"gpu_hit_rate:\s*([0-9.]+)", line)
                if match:
                    hit_rate = float(match.group(1))
                    break

        for line in reversed(lines):
            if "Average of first token latencies" in line:
                match = re.search(r"Average of first token latencies:\s*([0-9.]+)", line)
                if match:
                    latency = float(match.group(1))
                    break

        if cache_size not in results:
            results[cache_size] = {}

        results[cache_size][f"{cache_strategy.upper()}_hitrate"] = hit_rate
        results[cache_size][f"{cache_strategy.upper()}_time"] = latency

# 构建最终 DataFrame
df_rows = []
for cache_size in sorted(results.keys()):
    row = {"Cache_Size": cache_size}
    row.update(results[cache_size])
    df_rows.append(row)

df = pd.DataFrame(df_rows)

# 指定列顺序
columns = ["Cache_Size"]
for name in cache_strategies:
    name = name.upper()
    columns += [f"{name}_hitrate"]
for name in cache_strategies:
    name = name.upper()
    columns += [f"{name}_time"]

df = df[columns]

# 保存或打印结果
df.to_csv("./test/summary.csv", index=False)
print(df)
