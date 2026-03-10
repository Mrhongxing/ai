import torch  # 导入 PyTorch 库
from transformers import AutoTokenizer, AutoModelForCausalLM  # 导入 transformers 库中的自动分词器和因果语言模型
from peft import PeftModel  # 导入 peft 库中的 PeftModel

base_model = "/root/autodl-tmp/qwen-7B"  # 基础模型路径
lora_model = "/root/autodl-tmp/qwen-7b-lora"  # LoRA 微调模型路径

tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True
)  # 加载基础模型的分词器

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    trust_remote_code=True
)  # 加载基础模型的因果语言模型，自动分配设备

model = PeftModel.from_pretrained(
    model,
    lora_model,
    torch_dtype=torch.bfloat16
)  # 加载 LoRA 微调模型，使用 bfloat16 数据类型

prompt = "我最近总觉得一个人很孤单，孩子们也不常来看我。"  # 用户输入的提示语

messages = [
    {"role": "system", "content": "你是一位温和、有耐心、擅长与老年人沟通的心理支持助手。"},  # 系统设定角色
    {"role": "user", "content": prompt}  # 用户输入内容
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)  # 应用聊天模板，生成模型输入文本

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)  # 将文本转为张量，并移动到模型设备上

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)  # 生成模型输出，设置生成参数

print(tokenizer.decode(outputs[0], skip_special_tokens=True))  # 解码输出并打印，不显示特殊符号