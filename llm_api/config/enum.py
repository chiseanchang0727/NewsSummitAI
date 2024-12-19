from enum import Enum

class LLMType(Enum):
    
    OpenAIChat = 'openai_chat'
    AzureOpenAIChat = 'azure_openai_chat'
    
    def __repr__(self) -> str:
        # return as a string, like 'azure_openai_chat'
        return f"'{self.value}'"