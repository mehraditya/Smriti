"""
Single entry point for the LLM subsystem.
To add a new provider (e.g. Anthropic):
  1. Create core/llm/anthropic.py implementing BaseLLM
  2. Add "anthropic" to the match block
  3. Set LLM_PROVIDER=anthropic in .env
"""
import structlog

from config import get_settings
from core.llm.base import BaseLLM

log = structlog.get_logger()


def get_llm_engine() -> BaseLLM:
    settings = get_settings()
    provider = settings.llm_provider.lower()

    match provider:
        case "openai":
            from core.llm.openai import OpenAILLM
            return OpenAILLM()

        case "local":
            from core.llm.local import LocalLLM
            return LocalLLM()

        case _:
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Valid options: 'openai', 'local'. "
                f"Set LLM_PROVIDER in your .env file."
            )