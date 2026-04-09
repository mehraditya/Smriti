"""Embedding provider to implement this interface"""

import hashlib
from abc import ABC, abstractmethod

class BaseEmbedding(ABC):
    """
    Contract for all embedding providers
    """
    @abstractmethod
    async def embed(self, text: str) ->list[float]:
        """
        Embed a single string and return a float vector
        """

    @abstractmethod
    async def embed_batch(self, texts: list[str]) ->list[list[float]]:
        """
        Embed multiple strings, uses a single batched call. 
        Order of returned vectors must match input order
        """

    @property
    @abstractmethod
    def dimensions(self) ->int:
        """
        Vector dimensionality. Used at startup to 
        validate/create qdrant collection
        """

    @property
    @abstractmethod
    def provider_name(self) ->str:
        """Human-readable name to use in logs"""

    @staticmethod
    def hash_input(text: str) ->str:
        return hashlib.sha256(text.encode()).hexdigest()