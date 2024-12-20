
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

from fine_tuning.ft_configs import FineTuningConfig
from fine_tuning.data_preparation import process_dataset
from fine_tuning.utils import find_all_linear_names, count_parameters

def peft_fine_tuning(config: FineTuningConfig, dataset):
    device_map = {"": 0} if torch.cuda.is_available() else {"": "cpu"}

    model_name = config.model_name

    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=config.bnb_config,
        trust_remote_code=True,
        use_auth_token=True
    )

        
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', add_eos_token=True, add_bos_token=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    max_length = 2048
    process_data = process_dataset(tokenizer, max_length, dataset)  

    train_dataset, eval_dataset = process_data['train'].train_test_split(test_size=0.2).values()
    original_model = prepare_model_for_kbit_training(original_model)

    if not config.lora_config.target_modules:
        config.lora_config.target_modules = find_all_linear_names(model=original_model)

    # enable gradient checkpointing for memory efficiency
    original_model.gradient_checkpointing_enable()
    peft_model = get_peft_model(original_model, lora_config=config.lora_config)

    # show the percentage of trainable parameters
    count_parameters(model=peft_model)

    peft_model.use_cache = False

    peft_trainer = Trainer(
        model=peft_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=config.training_config,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    peft_trainer.train()


    peft_trainer.save_model('./results')


    torch.cuda.empty_cache()