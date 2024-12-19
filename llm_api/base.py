from llm_api.config.enum import LLMType
from langchain_openai import AzureChatOpenAI
from llm_api.config.llm_config import LLMConfig


class LLMBase:
    def __init__(self, config: LLMConfig):
        self.config = config
    
    def get_llm(self, llm_type=LLMType.AzureOpenAIChat):
        if llm_type == LLMType.AzureOpenAIChat:
            return  self.get_aoai_llm()
        else:
            raise NotImplementedError(f"LLM Type {llm_type} is not supported yet.")
        
    def get_aoai_llm(self):

        return AzureChatOpenAI(
            azure_endpoint=self.config.llm.api_base,
            openai_api_version=self.config.llm.api_version,
            azure_deployment=self.config.llm.deployment_name,
            openai_api_key=self.config.llm.api_key,
            temperature=self.config.llm.temperature
        )


    