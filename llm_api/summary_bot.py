from .base import LLMBase
from llm_api.config.llm_config import LLMConfig
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts.chat import ChatPromptTemplate

class SummaryBot(LLMBase):
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.model = self.get_llm()
       
    def summarize(self, input_string, prompt):
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", prompt
                ),
                (
                    "human","{input_string}"
                )
            ]
        )
        
        chain = (
            {"input_string": RunnablePassthrough()}
            | prompt
            | self.model
            | StrOutputParser()
        )
        
        return chain.invoke(input_string)