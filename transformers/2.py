from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline,BitsAndBytesConfig
""" quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # 启用4位量化
    llm_int8_enable_fp32_cpu_offload=False,
) """
tokenizer =AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")#使用AutoTokenizer加载预训练模型的分词器
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct",
    device_map="cpu",
    torch_dtype="auto",
    trust_remote_code=True,
    attn_implementation="eager"  # 添加这一行
    

)#使用AutoModelForCausalLM加载预训练模型的语言模型，并将其放置在MPS设备上以加速推理""" quantization_config=quantization_config,  # 添加这一行 """
generator = pipeline('text-generation', 
                     model=model, 
                     tokenizer=tokenizer,
                     return_full_text=False,
                     max_new_tokens=100,
                     use_cache=False,
                     do_sample=True)#使用pipeline创建一个文本生成器，指定模型和分词器，并设置生成文本的参数

response = generator("What is machine learning?")
print(response[0]['generated_text'])




input_text = "What is machine learning?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
output = model.model(input_ids)
print(output[0].shape)
lm_logits = output[0]
a=model.lm_head(lm_logits).shape
print(a)
b=a[0,-1].argmax(-1)
print(b)
c=tokenizer.decode(b)
print(c)