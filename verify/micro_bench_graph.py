import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取数据文件
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, 'result', 'statistics.txt')
output_dir = os.path.join(script_dir, 'result')

# 读取CSV数据
df = pd.read_csv(data_file)

# 创建图形
plt.figure(figsize=(12, 8))

# 定义颜色
colors = {
    'ARC': {'Miss': '#1f77b4', 'Hit': '#ff7f0e'},
    'LRU': {'Miss': '#2ca02c', 'Hit': '#d62728'}
}

# 为每个 EvictorType 绘制折线图
for evictor_type in df['EvictorType'].unique():
    data = df[df['EvictorType'] == evictor_type].sort_values('PromptLength')
    
    # 绘制 Miss_Mean
    plt.plot(data['PromptLength'], data['Miss_Mean'], 
             marker='o', linewidth=2, label=f'{evictor_type} Miss_Mean',
             color=colors[evictor_type]['Miss'])
    
    # 绘制 Hit_Mean
    plt.plot(data['PromptLength'], data['Hit_Mean'], 
             marker='s', linewidth=2, label=f'{evictor_type} Hit_Mean',
             color=colors[evictor_type]['Hit'])

# 设置图形属性
plt.xlabel('PromptLength', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.title('Micro Benchmark: Miss_TTFT and Hit_TTFT by EvictorType', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存第一张图片（包含 Miss 和 Hit）
output_file = os.path.join(output_dir, 'micro_bench_graph.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f'图表已保存到: {output_file}')

plt.close()

# 创建第二张图（只显示 Hit）
plt.figure(figsize=(12, 8))

# 为每个 EvictorType 绘制 Hit 折线图
for evictor_type in df['EvictorType'].unique():
    data = df[df['EvictorType'] == evictor_type].sort_values('PromptLength')
    
    # 只绘制 Hit_Mean
    plt.plot(data['PromptLength'], data['Hit_Mean'], 
             marker='s', linewidth=2, label=f'{evictor_type} Hit_Mean',
             color=colors[evictor_type]['Hit'])

# 设置图形属性
plt.xlabel('PromptLength', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.title('Micro Benchmark: Hit_TTFT by EvictorType', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存第二张图片（只显示 Hit）
output_file_hit = os.path.join(output_dir, 'micro_bench_graph_hit.png')
plt.savefig(output_file_hit, dpi=300, bbox_inches='tight')
print(f'图表已保存到: {output_file_hit}')

plt.close()

