"""
OpenAI chat completions. Works with any OpenAI-compatible endpoint
(Azure OpenAI, Together.ai, Groq, etc.) by pointing OPENAI_BASE_URL at the
alternative endpoint.
"""

import structlog
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from core.llm.base import BaseLLM

log = structlog.get_logger()


class OpenAILLM(BaseLLM):
    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,  # None = default OpenAI endpoint
        )
        self._model = settings.openai_chat_model
        log.info("llm.provider_init", provider="openai", model=self._model)

    @property
    def provider_name(self) -> str:
        return f"openai/{self._model}"

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""