import os
os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

model_path = "/root/autodl-tmp/qwen-7B"
train_file = "/root/autodl-tmp/elder_100k/shard_1093.jsonl"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16
)

model = prepare_model_for_kbit_training(model)

dataset = load_dataset(
    "json",
    data_files=train_file,
    split="train"
)

def format_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(format_chat, remove_columns=dataset.column_names)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="./qwen-7b-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=True,
    report_to="none",
    remove_unused_columns=False,
    load_best_model_at_end=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
    processing_class=tokenizer
)

trainer.train()
trainer.save_model("./qwen-7b-lora")