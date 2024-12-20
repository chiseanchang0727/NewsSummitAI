import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4', # normalized float 4
    bnb_4bit_compute_dtype=torch.bfloat16, # compute in float point 16
    bnb_4bit_use_double_quant=True # double quant to compress the model
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"":0},
    quantization_config=bnb_config
)
eval_tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=True, trust_remote_code=True, use_fast=False)
eval_tokenizer.pad_token = eval_tokenizer.eos_token
ft_model = PeftModel.from_pretrained(base_model, './results' ,torch_dtype=torch.float16, is_trainable=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ft_model.to(device)

input_text = ""
inputs = eval_tokenizer(input_text, return_tensors="pt").to(device)

# Generate outputs
with torch.no_grad():
    outputs = ft_model.generate(
        inputs.input_ids,
        max_length=50,  
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    
    
generated_text = eval_tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)