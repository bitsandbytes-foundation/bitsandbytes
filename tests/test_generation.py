import pytest
import torch
import math

from transformers import (
  AutoConfig,
  AutoModelForCausalLM,
  AutoTokenizer,
  BitsAndBytesConfig,
  GenerationConfig,
  set_seed,

)
import transformers


def get_4bit_config():
  return BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
  )


def get_model(model_name_or_path='huggyllama/llama-7b', bnb_config=get_4bit_config()):
  model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=bnb_config,
    max_memory={0:'48GB'},
    device_map='auto'
  ).eval()

  return model

def get_prompt_for_generation_eval(text, add_roles=True):
    description = (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    if add_roles:
        prompt = f'{description} ### Human: {text} ### Assistant:'
    else:
        prompt = f'{description} {text}'
    return prompt

def generate(model, tokenizer, text, generation_config, prompt_func=get_prompt_for_generation_eval):
    text = prompt_func(text)
    inputs = tokenizer(text, return_tensors="pt").to('cuda:0')
    outputs = model.generate(inputs=inputs['input_ids'], generation_config=generation_config)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

name_or_path = 'huggyllama/llama-7b'
#name_or_path = 'AI-Sweden/gpt-sw3-126m'

@pytest.fixture(scope='session')
def model():
    bnb_config = get_4bit_config()
    bnb_config.bnb_4bit_compute_dtype=torch.float32
    bnb_config.load_in_4bit=True
    model = get_model(name_or_path)
    print('')
    return model

@pytest.fixture(scope='session')
def tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained(name_or_path)
    return tokenizer

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=['fp16', 'bf16', 'fp32'])
def test_pi(model, tokenizer, dtype):

    generation_config = transformers.GenerationConfig(
        max_new_tokens=128,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )
    generation_config.max_new_tokens = 50


    #text = 'Please write down the first 50 digits of pi.'
    #text = get_prompt_for_generation_eval(text)
    #text += ' Sure, here the first 50 digits of pi: 3.14159'
    text = '3.14159'
    model.config.quantization_config.bnb_4bit_compute_dtype = dtype

    inputs = tokenizer(text, return_tensors="pt").to('cuda:0')
    outputs = model.generate(inputs=inputs['input_ids'], generation_config=generation_config)
    textout = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('')
    print(textout)
    print(math.pi)

    assert textout[:len(str(math.pi))] == str(math.pi)


