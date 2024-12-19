import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4', # normalized float 4
    bnb_4bit_compute_dtype=torch.bfloat16, # compute in float point 16
    bnb_4bit_use_double_quant=True # double quant to compress the model
)


lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
        'o_proj',
        'gate_proj',
        'up_proj',
        'down_proj'
    ],
    bias='none',
    lora_dropout=0.05,
    task_type='CAUSAL_LM'
)
