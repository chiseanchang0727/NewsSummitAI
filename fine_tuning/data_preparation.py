from functools import partial
from datasets import Dataset
from utils.utils import MySQLAgent
from src.queries.queries import GET_NEWS_INPUT
from utils.utils import get_db_data



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

def process_dataset(tokenizer, max_length, query):

    data = get_db_data(query)

    dataset = Dataset.from_list(data)

    dataset = dataset.map(create_llama_prompt_formats)
    process_func = partial(process_batch_data, tokenizer=tokenizer, max_length=max_length)
    dataset = dataset.map(
        process_func, batched=True, remove_columns=['input', 'output', 'instruction']
    )
    
    dataset = dataset.shuffle(seed=6)
    
    return dataset



