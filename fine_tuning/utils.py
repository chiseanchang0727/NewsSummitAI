from typing import Type, TypeVar
from pydantic import BaseModel
import yaml
import bitsandbytes as bnb


T = TypeVar('T', bound=BaseModel)

def load_config_from_yaml(file_path: str, model: Type[T]) -> T:
    """
    Load a YAML configuration file and parse it into a specified Pydantic model.

    Args:
        file_path (str): The path to the YAML file.
        model (Type[T]): The Pydantic model class to parse the data into.

    Returns:
        T: An instance of the specified Pydantic model.
    """
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return model(**yaml_data['Configs'])


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


def count_parameters(model):
    all_para = sum(p.numel() for p in model.parameters())
    trainable_para = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('all model parameters:', all_para)
    print('trainable model parameters:', trainable_para)

    print(f'percentage of trainable parameters: {trainable_para / all_para * 100:.2f}%')