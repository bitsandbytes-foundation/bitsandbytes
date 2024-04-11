import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time

MAX_NEW_TOKENS = 64
model_id = "facebook/opt-1.3b"

text = 'Hamburg is in which country?\n'
tokenizer = AutoTokenizer.from_pretrained(model_id)
input_ids = tokenizer(text, return_tensors="pt").input_ids

print('Loading model {}...'.format(model_id))
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
  model_id,
  device_map='auto',
  quantization_config=quantization_config,
  torch_dtype=torch.bfloat16
)
print('model dtype = {}'.format(model.dtype))

with torch.no_grad():
    t0 = time.time()
    generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)
    latency = time.time() - t0
    result = "| latency: " + str(round(latency * 1000, 3)) + " ms |"
    print('+' + '-' * (len(result) - 2) + '+')
    print(result)
    print('+' + '-' * (len(result) - 2) + '+')

output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(f"output: {output}")
