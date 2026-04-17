"""
Local LLM via Ollama (https://ollama.com).
Ollama exposes an OpenAI-compatible /v1/chat/completions endpoint,
so we reuse the AsyncOpenAI client pointed at localhost.

Setup:
  1. Install Ollama: https://ollama.com/download
  2. Pull a model: ollama pull llama3.2 (or mistral, phi3, gemma2, etc.)
  3. Set in .env:
       LLM_PROVIDER=local
       LOCAL_LLM_MODEL=llama3.2
       LOCAL_LLM_BASE_URL=http://localhost:11434/v1
Note: Local models vary in JSON reliability. The ingestion engine already
handles parse failures gracefully, but for production use you'll want to
run evals on extraction quality before switching from OpenAI.
"""
import structlog
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from core.llm.base import BaseLLM

log = structlog.get_logger()


class LocalLLM(BaseLLM):
    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.local_llm_model
        self._base_url = settings.local_llm_base_url
        # Ollama's OpenAI-compat endpoint doesn't validate the key,
        # but the client requires a non-empty string
        self._client = AsyncOpenAI(
            api_key="ollama",
            base_url=self._base_url,
        )
        log.info(
            "llm.provider_init",
            provider="local/ollama",
            model=self._model,
            base_url=self._base_url,
        )

    @property
    def provider_name(self) -> str:
        return f"local/{self._model}"

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=10))
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