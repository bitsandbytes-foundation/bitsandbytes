from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1b-Instruct")
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1b-Instruct", device_map="mps", quantization_config=quantization_config)
print("model.device:", model.device)
prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))  # or whatever entry function you have