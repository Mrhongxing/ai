import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "/root/autodl-tmp/qwen-7B"
lora_model = "/root/autodl-tmp/qwen-7b-lora"

tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(
    model,
    lora_model,
    torch_dtype=torch.bfloat16
)

prompt = "我最近总觉得一个人很孤单，孩子们也不常来看我。"

messages = [
    {"role": "system", "content": "你是一位温和、有耐心、擅长与老年人沟通的心理支持助手。"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))