import os
import re
import pandas as pd

def calculate_hitrate_after_cold_start(lines):
    """
    计算冷启动后的命中率
    
    Args:
        lines: 日志文件的所有行
        
    Returns:
        float or None: 冷启动后命中率，如果无法计算则返回None
    """
    # 找到[cold start]行的位置
    cold_start_line_idx = None
    for i, line in enumerate(lines):
        if "[cold start]" in line:
            cold_start_line_idx = i
            break
    
    if cold_start_line_idx is None:
        return None
    
    # 冷启动前的最后一次hit和access数据
    hit1 = None
    access1 = None
    for i in range(cold_start_line_idx - 1, -1, -1):
        if "hit:" in lines[i] and "access:" in lines[i]:
            match_hit = re.search(r"hit:\s*([0-9]+)", lines[i])
            match_access = re.search(r"access:\s*([0-9]+)", lines[i])
            if match_hit and match_access:
                hit1 = int(match_hit.group(1))
                access1 = int(match_access.group(1))
                break
    
    # 整个文件最后一次hit和access数据
    hit2 = None
    access2 = None
    for line in reversed(lines):
        if "hit:" in line and "access:" in line:
            match_hit = re.search(r"hit:\s*([0-9]+)", line)
            match_access = re.search(r"access:\s*([0-9]+)", line)
            if match_hit and match_access:
                hit2 = int(match_hit.group(1))
                access2 = int(match_access.group(1))
                break
    
    # 计算冷启动后的命中率
    if hit1 is not None and access1 is not None and hit2 is not None and access2 is not None:
        if access2 - access1 > 0:
            return (hit2 - hit1) / (access2 - access1)
    
    return None

# 设置变量
model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
dataset_name = "Quality"   # wild
sample_strategy = "Distshift" # all
cache_strategies = ["arc", "dbl", "lru"]
# cache_strategies = ["lru", "arc"]
cache_strategies = ["dbl"]

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

        # 找出最后一次出现的 gpu_hit_rate 和两种 latency
        hit_rate = None
        latency_total = None
        latency_after_cold_start = None
        hitrate_after_cold_start = calculate_hitrate_after_cold_start(lines)
        
        for line in reversed(lines):
            if "gpu_hit_rate" in line:
                match = re.search(r"gpu_hit_rate:\s*([0-9.]+)", line)
                if match:
                    hit_rate = float(match.group(1))
                    break

        # 找出最后一次出现的 "Average of first token latencies (after cold start):"
        for line in reversed(lines):
            if "Average of first token latencies (after cold start):" in line:
                match = re.search(r"Average of first token latencies \(after cold start\):\s*([0-9.]+)", line)
                if match:
                    latency_after_cold_start = float(match.group(1))
                    break

        # 找出最后一次出现的 "Average of first token latencies:" (不包含after cold start)
        for line in reversed(lines):
            if "Average of first token latencies:" in line and "(after cold start)" not in line:
                match = re.search(r"Average of first token latencies:\s*([0-9.]+)", line)
                if match:
                    latency_total = float(match.group(1))
                    break

        if cache_size not in results:
            results[cache_size] = {}

        results[cache_size][f"{cache_strategy.upper()}_hitrate"] = hit_rate
        results[cache_size][f"{cache_strategy.upper()}_time_total"] = latency_total
        results[cache_size][f"{cache_strategy.upper()}_hitrate_after_cold"] = hitrate_after_cold_start
        results[cache_size][f"{cache_strategy.upper()}_time_after_cold"] = latency_after_cold_start

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
    columns += [f"{name}_time_total"]
for name in cache_strategies:
    name = name.upper()
    columns += [f"{name}_hitrate_after_cold"]
for name in cache_strategies:
    name = name.upper()
    columns += [f"{name}_time_after_cold"]

df = df[columns]

# 保存或打印结果
df.to_csv("./test/summary.csv", index=False)
print(df)
