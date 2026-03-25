import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/Users/mac/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct/snapshots/f39ac1d28e925b323eae81227eaba4464caced4e"

print("loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)

messages = [
    {"role": "user", "content": "What is machine learning?"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt")

print("start generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        use_cache=True
    )

print("generation finished")

new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
result = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("model output:")
print(result)