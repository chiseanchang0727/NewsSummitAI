from pydantic import BaseModel, Field
from typing import List
from typing import Optional

class BnbConfig(BaseModel):
    load_in_4bit: bool = Field(default=None, description="Whether to load the model in 4-bit mode")
    bnb_4bit_quant_type: str = Field(default=None, description="Quantization type for 4-bit")
    bnb_4bit_compute_dtype: str = Field(default=None, description="Compute dtype for 4-bit mode")
    bnb_4bit_use_double_quant: bool = Field(default=None, description="Whether to use double quantization in 4-bit mode")

class LoraConfig(BaseModel):
    r: int = Field(default=None, description="LoRA rank")
    lora_alpha: int = Field(default=None, description="LoRA alpha")
    target_modules: List[str] = Field(default=None, description="List of target modules")
    bias: str = Field("none", description="Type of bias used")
    lora_dropout: float = Field(0.0, description="Dropout rate for LoRA layers")
    task_type: str = Field(default=None, description="Type of task (e.g., CAUSAL_LM)")


class TrainingArgumentsConfig(BaseModel):
    output_dir: str = Field(default=None, description="Directory to save training results")
    warmup_steps: int = Field(default=None, description="Number of warmup steps for learning rate scheduler")
    per_device_train_batch_size: int = Field(default=None, description="Batch size per device during training")
    gradient_accumulation_steps: int = Field(default=None, description="Number of steps to accumulate gradients")
    max_steps: int = Field(default=None, description="Total number of training steps")
    learning_rate: float = Field(default=None, description="Initial learning rate for optimizer")
    optim: str = Field(default=None, description="Optimizer to use (e.g., 'adamw', 'paged_adamw_8bit')")
    logging_steps: int = Field(default=None, description="Number of steps between logging outputs")
    logging_dir: str = Field(default=None, description="Directory to save logs")
    save_strategy: str = Field(default=None, description="Saving strategy (e.g., 'steps', 'epoch')")
    save_steps: int = Field(default=None, description="Number of steps between saving checkpoints")
    evaluation_strategy: str = Field(default=None, description="Evaluation strategy during training")
    do_eval: bool = Field(default=None, description="Whether to perform evaluation")
    gradient_checkpointing: bool = Field(default=None, description="Whether to use gradient checkpointing")
    report_to: Optional[str] = Field(None, description="Integration to report metrics to (e.g., 'wandb', 'tensorboard')")
    overwrite_output_dir: bool = Field(default=None, description="Whether to overwrite the output directory")
    group_by_length: bool = Field(default=None, description="Whether to group sequences by length during training")


class FineTuningConfig(BaseModel):
    model_name: str = Field(default=None, description="Name of the model")
    bnb_config: BnbConfig = Field(default_factory=BnbConfig, description="Bit and byte quantization configuration")
    lora_config: LoraConfig = Field(default_factory=LoraConfig, description="LoRA configuration")
    training_config: TrainingArgumentsConfig = Field(default_factory=TrainingArgumentsConfig, description="Training configs.")