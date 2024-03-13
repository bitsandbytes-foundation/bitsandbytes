import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

MAX_NEW_TOKENS = 128
model_name = "meta-llama/Llama-2-7b-hf"

text = "Hamburg is in which country?\n"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
input_ids = tokenizer(text, return_tensors="pt").input_ids

max_memory = f"{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB"

n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}

model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True, max_memory=max_memory)

generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
