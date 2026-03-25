import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

model_path = "/Users/mac/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct/snapshots/f39ac1d28e925b323eae81227eaba4464caced4e"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

print("loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device == "mps" else torch.float32,
    trust_remote_code=True,
    attn_implementation="eager"
).to(device)

messages = [
    {"role": "user", "content": "What is machine learning?"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

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