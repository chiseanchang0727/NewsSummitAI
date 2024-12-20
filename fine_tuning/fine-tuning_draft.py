from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from functools import partial


torch.cuda.is_available()


model_name = 'shenzhi-wang/Llama3.1-8B-Chinese-Chat'
device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}


# Set qunatization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4', # normalized float 4
    bnb_4bit_compute_dtype=torch.bfloat16, # compute in float point 16
    bnb_4bit_use_double_quant=True # double quant to compress the model
)

# Passing device_map = 0 means put the whole model on GPU 0
original_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_auth_token=True
)


tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', add_eos_token=True, add_bos_token=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# Process dataset

# faq_datasets = [{'instruction':instruction ,'input':item[0]['question'], 'output':'faq'} for item in faq_json.values()]



def create_llama_prompt_formats(sample):

    instruction = sample['instruction']
    input_text = sample['input']
    output_text = sample['output']
    
    formatted_prompt = f"<s>[INST] <<SYS>>\n{instruction}\n<</SYS>>\n{input_text}\n[/INST] {output_text} </s>"
    sample['text'] = formatted_prompt
    
    return sample

def process_batch_data(batch, tokenizer, max_length):
    return tokenizer(
        batch['text'],
        max_length=max_length,
        truncation=True
    )

def process_dataset(tokenizer, max_length, data):

    dataset = Dataset.from_list(data)

    dataset = dataset.map(create_llama_prompt_formats)
    process_func = partial(process_batch_data, tokenizer=tokenizer, max_length=max_length)
    dataset = dataset.map(
        process_func, batched=True, remove_columns=['input', 'output', 'instruction']
    )
    
    dataset = dataset.shuffle(seed=6)
    
    return dataset

max_length = 2048
process_data = process_dataset(tokenizer, max_length, dataset)

train_dataset, eval_dataset = process_data['train'].train_test_split(test_size=0.2).values()

original_model = prepare_model_for_kbit_training(original_model)


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

# enable gradient checkpointing for memory efficiency
original_model.gradient_checkpointing_enable()
peft_model = get_peft_model(original_model, lora_config)

def count_parameters(model):
    all_para = sum(p.numel() for p in model.parameters())
    trainable_para = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('all model parameters:', all_para)
    print('trainable model parameters:', trainable_para)

    print(f'percentage of trainable parameters: {trainable_para / all_para * 100:.2f}%')

# count_parameters(peft_model)

peft_training_args = TrainingArguments(
    output_dir= 'results',
    warmup_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=10,
    learning_rate=2e-4,
    optim='paged_adamw_8bit',
    logging_steps=25,
    logging_dir='./logs',
    save_strategy='steps',
    save_steps=25,
    evaluation_strategy='steps',
    do_eval=True,
    gradient_checkpointing=True,
    report_to='none',
    overwrite_output_dir=True,
    group_by_length=True,
    
)

peft_model.use_cache = False

peft_trainer = Trainer(
    model=peft_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=peft_training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)


peft_trainer.train()


peft_trainer.save_model('./results')


torch.cuda.empty_cache()