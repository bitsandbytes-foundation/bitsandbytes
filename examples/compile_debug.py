import logging

import torch
import torch._dynamo
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

torch._logging.set_logs(
    dynamo=logging.INFO,
    graph_breaks=True,
    recompiles=True,
    recompiles_verbose=True,
    compiled_autograd_verbose=True,
)

torch._dynamo.config.suppress_errors = False


torch.set_float32_matmul_precision("high")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# torch._dynamo.config.capture_dynamic_output_shape_ops = True

# model_id = "google/gemma-2-2b-it"
model_id = "Qwen/Qwen2.5-7B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

compile_options = {
    # "epilogue_fusion": True,
    # "shape_padding": True,
    # "trace.enabled"     : True,
    # "triton.cudagraphs" : False,
}

# warmup
outputs = model.generate(**input_ids, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))

# compile

model.forward = torch.compile(model.forward, dynamic=True, fullgraph=True, options=compile_options)

# model = torch.compile(model, dynamic=True, fullgraph=True, options=compile_options)

outputs = model.generate(**input_ids, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
