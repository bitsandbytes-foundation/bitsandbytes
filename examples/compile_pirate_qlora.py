# for torch.compile trace run:
#
# TORCH_TRACE="./tracedir" TORCH_LOGS="graph_breaks" CUDA_VISIBLE_DEVICES=0 python examples/compile_pirate_qlora.py

# 🏴☠️⛵ Pirate Coder's Delight: Fine-Tune Mistral-7B to Speak Like a Buccaneer Hacker
# Using bitsandbytes 4-bit + torch.compile - 100% Original Dataset

from datasets import Dataset
from peft import LoraConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# 1. Load Model with Pirate-Optimized Quantization 🏴☠️
model_id = "Qwen/Qwen1.5-1.8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "bnb_4bit_use_double_quant": True,
    },
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)


# 2. Original Pirate Programmer Dataset 🦜
def pirate_formatting_func(example):
    return {"text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"}


train_dataset = Dataset.from_list(
    [
        {
            "instruction": "Explain quantum computing using pirate slang",
            "response": "Arrr, matey! 'Tis like sailin' parallel seas...",
        },
        {
            "instruction": "Write Python code to find buried treasure",
            "response": "def find_booty():\n    return (sum(coordinates) / len(coordinates))",
        },
        {
            "instruction": "Why do pirates hate distributed systems?",
            "response": "Too many captains sink the ship, ye scallywag!",
        },
    ]
).map(pirate_formatting_func)


# 2. Prepare Pirate Dataset
def tokenize_pirate_data(examples):
    return tokenizer(examples["text"], padding="max_length", max_length=256, truncation=True, return_tensors="pt")


train_dataset = train_dataset.map(
    tokenize_pirate_data,
    batched=True,
    remove_columns=["instruction", "response"],  # Keep only tokenized fields
)

# 3. Configure QLoRA ⚙️
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. Training Configuration
training_args = SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    max_steps=5,
    learning_rate=2e-5,
    max_seq_length=256,
    remove_unused_columns=False,
    output_dir="./pirate_coder",
    optim="paged_adamw_8bit",
    dataset_text_field="text",
    packing=True,
    torch_compile={
        "mode": "reduce-overhead",
        "fullgraph": False,
        "dynamic": False,
    },
    report_to="none",
    logging_steps=1,
)

# 6. Launch Training with Pirate Flair! 🚀
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    peft_config=peft_config,
    formatting_func=pirate_formatting_func,
)

print("⚡ Batten down the hatches - training with torch.compile!")
trainer.train()
