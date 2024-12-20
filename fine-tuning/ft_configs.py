import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig
import bitsandbytes as bnb

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
    bias='none', # Specifies whether and how biases are handled in LoRA.
    lora_dropout=0.05,
    task_type='CAUSAL_LM'
)






def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            # model-specific
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)