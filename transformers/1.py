from transformers import AutoTokenizer

# 定义颜色列表（RGB 值）
colors = [
    "38;2;255;0;0",    # 红色
    "38;2;0;255;0",    # 绿色
    "38;2;0;0;255",    # 蓝色
    "38;2;255;255;0",  # 黄色
    "38;2;255;0;255",  # 品红
    "38;2;0;255;255",  # 青色
]

def show_tokens(sentence: str, name: str):
    """显示每个分词结果，用不同颜色分隔"""
    
    # 加载 tokenizer 并分词
    tokenizer = AutoTokenizer.from_pretrained(name)
    token_ids = tokenizer(sentence).input_ids
    
    # 打印词汇表大小
    print(f"Vocab length: {len(tokenizer)}")
    print(f"Original text: {sentence}")
    print("Tokens:")
    
    # 打印带颜色的 token
    for idx, t in enumerate(token_ids):
        # ANSI 转义码：前景色，背景色透明
        color_code = f"\033[1;{colors[idx % len(colors)]}m"
        reset_code = "\033[0m"
        
        # 解码 token
        token_text = tokenizer.decode(t)
        
        # 打印带颜色的 token
        print(f"{color_code}{token_text}{reset_code}", end=' ')
    
    print()  # 换行
    print("-" * 50)

# 使用示例
""" if __name__ == "__main__":
    show_tokens("Hello, I'm learning transformers!", "bert-base-uncased")
    show_tokens("你好，我在学习深度学习！", "bert-base-chinese")
from transformers import AutoTokenizer, AutoModelForCausalLM"""
a= "Hello, how are you?" 

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
show_tokens(a, "Xenova/gpt-4")
token_ids = tokenizer(a).input_ids
print(token_ids)
for token in token_ids:
    print(tokenizer.decode(token))