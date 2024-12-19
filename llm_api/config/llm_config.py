# from devtools import pformat

from pydantic import BaseModel, Field

from llm_api.config.enum import LLMType
from llm_api.config.llm_parameters import LLMParameters

class LLMConfig(BaseModel):
    
    # def __repr__(self) -> str:
    #     """Get a string representation."""
    #     return pformat(self, highlight=False)
    
    type: LLMType = Field(
        default=LLMType.AzureOpenAIChat,
        description='The type of LLM model to sue'
    )
    
    llm: LLMParameters = Field(
        default_factory=LLMParameters,
        description="The LLM configuration to use."
    )