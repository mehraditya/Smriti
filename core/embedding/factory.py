"""
Single entry point for the embedding subsystem
The rest of the codebase only imports from here - never from local.py or others
"""

import structlog

from config import get_settings
from core.embedding.base import BaseEmbedding

log= structlog.get_logger()

def get_embedding_engine()-> BaseEmbedding:
    """
    Instantiate and return the embedding provider configured in settings.
    called once at app startup
    """

    settings = get_settings()
    provider = settings.embedding_provider.lower()

    match provider:
        case "openai":
            from core.embedding.openai import OpenAIEmbedding
            return OpenAIEmbedding()
        
        case "local":
            from core.embedding.local import LocalEmbedding
            return LocalEmbedding()
        
        case _:
            raise ValueError(
                f"Unknown embedding provider: '{provider}.'"
            )