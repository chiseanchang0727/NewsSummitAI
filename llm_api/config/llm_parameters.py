import os
from pydantic import BaseModel, Field

class LLMParameters(BaseModel):
    """LLM Parameters model."""
    
    ############################ 
    # Load from .env by default
    api_key: str = Field(
        description="The API key to use for the LLM service.",
        default_factory=lambda: os.getenv("OPENAI_API_KEY"), 
    )
    
    api_base: str | None = Field(
        description="The base URL for the LLM API.",
        default_factory=lambda: os.getenv("OPENAI_API_BASE"), 
    )
    
    api_version: str | None = Field(
        description="The version of the LLM API to use.",
        default_factory=lambda: os.getenv("OPEN_AI_VERSION"), 
    )
    
    deployment_name: str | None = Field(
        description="The deployment name to use for the LLM service.",
        default_factory=lambda: os.getenv("GPT_DEPLOYMENT_NAME"),  
    )
    ############################
    
    temperature: float | None = Field(
        description="The temperature to use for token generation.",
        default=0,
    )

