import pandas as pd
import numpy as np
from pathlib import Path

# 读取数据文件
data_file = Path(__file__).parent / 'result' / 'microbench_300.txt'
df = pd.read_csv(data_file)

# 过滤掉空行
df = df.dropna()

# 按 EvictorType 和 PromptLength 分组
grouped = df.groupby(['EvictorType', 'PromptLength'])

# 计算统计信息
results = []
for (evictor_type, prompt_length), group in grouped:
    miss_values = group['Miss'].values
    hit_values = group['Hit'].values
    
    miss_mean = np.mean(miss_values)
    miss_std = np.std(miss_values, ddof=1)  # 使用样本标准差 (n-1)
    hit_mean = np.mean(hit_values)
    hit_std = np.std(hit_values, ddof=1)
    
    results.append({
        'EvictorType': evictor_type,
        'PromptLength': prompt_length,
        'Miss_Mean': miss_mean,
        'Miss_Std': miss_std,
        'Hit_Mean': hit_mean,
        'Hit_Std': hit_std,
        'Num_Runs': len(group)
    })

# 创建结果 DataFrame
result_df = pd.DataFrame(results)

# 打印结果
print("=" * 80)
print("统计结果：每种 EvictorType 和 PromptLength 组合的 Miss 和 Hit 平均值与标准差")
print("=" * 80)
print()
print(result_df.to_string(index=False))
print()

# 也可以保存到文件
output_file = Path(__file__).parent / 'result' / 'statistics.txt'
result_df.to_csv(output_file, index=False, float_format='%.6f')
print(f"结果已保存到: {output_file}")

