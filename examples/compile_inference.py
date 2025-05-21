import torch
import torch._dynamo
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# torch._dynamo.config.suppress_errors = True

torch.set_float32_matmul_precision("high")

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# torch._dynamo.config.capture_dynamic_output_shape_ops = True

model_id = "google/gemma-2-2b-it"
# model_id = "Qwen/Qwen2.5-7B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

# model.forward = torch.compile(model.forward, fullgraph=True)

model = torch.compile(model)

outputs = model.generate(**input_ids, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
