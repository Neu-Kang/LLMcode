from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = 'G:/Pycode/Meta-Llama-3___1-8B-Instruct'
lora_path = 'G:/Pycode/self-llm-master/examples/Chat-嬛嬛/output/llama3_1_instruct_lora/checkpoint-699'

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

prompt = "嬛嬛你怎么了，朕替你打抱不平！"

messages = [
        {"role": "system", "content": "假设你是皇帝身边的女人--甄嬛。"},
        {"role": "user", "content": prompt}
]

input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# print(input_ids)

model_inputs = tokenizer([input_ids], return_tensors="pt")

# 确保model_inputs转移到与模型相同的设备上
model_inputs = model_inputs.to(model.device)

generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print('皇上：', prompt)
print('嬛嬛：', response)

