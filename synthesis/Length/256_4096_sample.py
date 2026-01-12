import json
import random
from typing import List

# 字母表：b~z (共25个字母)
LETTERS = [chr(i) for i in range(ord('b'), ord('z') + 1)]

def generate_block(block_id: int) -> str:
    """
    生成一个16字符的block
    通过不同的排列组合生成独一无二的block
    按照顺序生成：第一个block全是b，第二个block是15个b+1个c，等等
    """
    # 将block_id转换为25进制的16位数字
    # 这样可以生成25^16种不同的block组合
    block = []
    remaining = block_id
    for i in range(16):
        letter_idx = remaining % 25
        block.append(LETTERS[letter_idx])
        remaining = remaining // 25
    return ''.join(block)

def generate_sentence(target_length: int, sentence_id: int) -> str:
    """
    生成一个指定长度的句子
    target_length: 目标长度（允许±16的误差）
    sentence_id: 句子ID，用于生成不同的block组合
    """
    # 使用sentence_id作为随机种子，确保每个句子ID生成唯一的句子
    rng = random.Random(sentence_id)
    
    # 计算需要多少个block
    num_blocks = target_length // 16
    # 生成blocks
    blocks = []
    for i in range(num_blocks):
        # 使用sentence_id和block位置来生成唯一的block
        # 使用更大的乘数确保不同句子使用不同的block范围
        block_id = sentence_id * 10000 + i
        block = generate_block(block_id)
        blocks.append(block)
    
    # 组合所有blocks
    sentence = ''.join(blocks)
    
    # 添加随机误差（±16），使用rng确保可复现
    length_error = rng.randint(-16, 16)
    actual_length = target_length + length_error
    actual_length = max(1, actual_length)  # 确保至少1个字符
    
    # 如果实际长度与目标长度不同，调整句子长度
    if len(sentence) < actual_length:
        # 如果太短，添加随机字符
        extra = actual_length - len(sentence)
        sentence += ''.join(rng.choices(LETTERS, k=extra))
    elif len(sentence) > actual_length:
        # 如果太长，截断
        sentence = sentence[:actual_length]
    
    # 将句子转换为空格分隔的格式（每个字符后加空格）
    return ' '.join(sentence)

def generate_all_sentences(num_256: int = 500, num_4096: int = 5):
    """
    生成所有需要的句子
    """
    # 生成 num_256 种长度为256的句子
    sentences_256 = []
    for i in range(num_256):
        sentence = generate_sentence(256, i)
        sentences_256.append(sentence)
    
    # 生成 num_4096 种长度为4096的句子
    sentences_4096 = []
    for i in range(num_4096):
        sentence = generate_sentence(4096, i)
        sentences_4096.append(sentence)
    
    return sentences_256, sentences_4096

def sample_requests(
    sentences_256: List[str],
    sentences_4096: List[str],
    num_requests: int = 3072,
    prob_256: float = 0.862,
):
    """
    采样request
    86.2%的概率选择256长度的句子，13.8%的概率选择4096长度的句子
    """
    requests = []
    for i in range(num_requests):
        # 根据概率选择长度
        if random.random() < prob_256:
            # 选择256长度的句子
            sentence = random.choice(sentences_256)
        else:
            # 选择4096长度的句子
            print(f"选择4096长度的句子", i)
            sentence = random.choice(sentences_4096)
        requests.append(sentence)
    return requests

def main():
    # 设置随机种子，确保结果可复现
    random.seed(42)
    
    # 当前需求：2000 个 256 长度句子，5 个 4096 长度句子
    num_256 = 2000
    num_4096 = 5
    prob_256 = num_256 / (num_4096 + num_256)
    num_requests = 4096

    print("生成句子...")
    sentences_256, sentences_4096 = generate_all_sentences(
        num_256=num_256, num_4096=num_4096
    )
    print(f"生成了 {len(sentences_256)} 个256长度的句子")
    print(f"生成了 {len(sentences_4096)} 个4096长度的句子")
    
    # 验证句子长度
    print("\n验证句子长度...")
    lengths_256 = [len(s.replace(' ', '')) for s in sentences_256]
    lengths_4096 = [len(s.replace(' ', '')) for s in sentences_4096]
    print(f"256长度句子范围: {min(lengths_256)} - {max(lengths_256)}")
    print(f"4096长度句子范围: {min(lengths_4096)} - {max(lengths_4096)}")
    
    print("\n采样requests...")
    requests = sample_requests(
        sentences_256,
        sentences_4096,
        num_requests=num_requests,
        prob_256=prob_256,
    )
    print(f"采样了 {len(requests)} 个requests")
    
    # 统计采样结果
    count_256 = sum(1 for r in requests if len(r.replace(' ', '')) < 3000)
    count_4096 = len(requests) - count_256
    print(f"256长度句子数量: {count_256} ({count_256/len(requests)*100:.1f}%)")
    print(f"4096长度句子数量: {count_4096} ({count_4096/len(requests)*100:.1f}%)")
    
    print("\n写入文件...")
    with open('sentences.json', 'w', encoding='utf-8') as f:
        json.dump(requests, f, indent=4, ensure_ascii=False)
    
    print("完成！")

if __name__ == '__main__':
    main()

