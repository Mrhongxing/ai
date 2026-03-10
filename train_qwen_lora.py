import os  # 导入操作系统模块
os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"  # 设置混合精度为bf16

import torch  # 导入PyTorch库
from datasets import load_dataset  # 导入datasets库中的load_dataset函数
from transformers import (  # 导入transformers库中的相关类和函数
    AutoTokenizer,  # 自动分词器
    AutoModelForCausalLM,  # 自动因果语言模型
    BitsAndBytesConfig,  # 量化配置
    TrainingArguments  # 训练参数
)
from peft import LoraConfig, prepare_model_for_kbit_training  # 导入peft库中的Lora配置和模型准备函数
from trl import SFTTrainer  # 导入trl库中的SFTTrainer训练器

model_path = "/root/autodl-tmp/qwen-7B"  # 模型路径
train_file = "/root/autodl-tmp/elder_100k/shard_1093.jsonl"  # 训练数据文件路径

tokenizer = AutoTokenizer.from_pretrained(  # 加载预训练分词器
    model_path,
    trust_remote_code=True  # 信任远程代码
)

if tokenizer.pad_token is None:  # 如果分词器没有pad_token
    tokenizer.pad_token = tokenizer.eos_token  # 设置pad_token为eos_token

bnb_config = BitsAndBytesConfig(  # 创建量化配置
    load_in_4bit=True,  # 使用4bit量化加载
    bnb_4bit_use_double_quant=True,  # 使用双重量化
    bnb_4bit_quant_type="nf4",  # 量化类型为nf4
    bnb_4bit_compute_dtype=torch.bfloat16  # 计算数据类型为bfloat16
)

model = AutoModelForCausalLM.from_pretrained(  # 加载预训练因果语言模型
    model_path,
    device_map="auto",  # 自动分配设备
    trust_remote_code=True,  # 信任远程代码
    quantization_config=bnb_config,  # 使用量化配置
    torch_dtype=torch.bfloat16  # 数据类型为bfloat16
)

model = prepare_model_for_kbit_training(model)  # 准备模型进行kbit训练

dataset = load_dataset(  # 加载数据集
    "json",  # 数据格式为json
    data_files=train_file,  # 数据文件路径
    split="train"  # 加载训练集
)

def format_chat(example):  # 定义格式化聊天数据的函数
    text = tokenizer.apply_chat_template(  # 应用聊天模板生成文本
        example["messages"],
        tokenize=False,  # 不进行分词
        add_generation_prompt=False  # 不添加生成提示
    )
    return {"text": text}  # 返回格式化后的文本

dataset = dataset.map(format_chat, remove_columns=dataset.column_names)  # 映射格式化函数并移除原有列

lora_config = LoraConfig(  # 创建Lora配置
    r=16,  # Lora秩为16
    lora_alpha=32,  # Lora alpha为32
    lora_dropout=0.05,  # Lora dropout为0.05
    bias="none",  # 不使用偏置
    task_type="CAUSAL_LM"  # 任务类型为因果语言模型
)

training_args = TrainingArguments(  # 创建训练参数
    output_dir="./qwen-7b-lora",  # 输出目录
    per_device_train_batch_size=2,  # 每设备训练批量大小为2
    gradient_accumulation_steps=8,  # 梯度累积步数为8
    learning_rate=2e-4,  # 学习率为2e-4
    num_train_epochs=3,  # 训练轮数为3
    logging_steps=10,  # 日志记录步数为10
    save_strategy="epoch",  # 保存策略为每轮保存
    fp16=False,  # 不使用fp16
    bf16=True,  # 使用bf16
    report_to="none",  # 不报告到任何平台
    remove_unused_columns=False,  # 不移除未使用的列
    load_best_model_at_end=False  # 不在训练结束时加载最佳模型
)

trainer = SFTTrainer(  # 创建SFTTrainer训练器
    model=model,  # 使用的模型
    train_dataset=dataset,  # 训练数据集
    args=training_args,  # 训练参数
    peft_config=lora_config,  # Lora配置
    processing_class=tokenizer  # 处理类为分词器
)

trainer.train()  # 开始训练
trainer.save_model("./qwen-7b-lora")  # 保存模型到指定目录