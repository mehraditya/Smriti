"""
Every LLM provider implements this interface
The ingestion engine uses BaseLLM
"""

from abc import ABC, abstractmethod

class BaseLLM(ABC):
    # Contract for chat completion providers used in mem-extraction and relation classification

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) ->str:
        """
        Send a system + user message pair and return the LLM text response
        Implementation handles retries, timeouts, and provider specific errors internally
        """

    @property
    @abstractmethod
    def provider_name(self) ->str:
        


        
        