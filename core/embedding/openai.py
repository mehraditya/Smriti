"""
OpenAI embeddings via official async client
use text-embedding-3-small
"""

import structlog

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from core.embedding.base import BaseEmbedding

log = structlog.get_logger()

# Dimensions for known OpenAI embedding models.
# text-embedding-3-* support shortening via the `dimensions` param,
# but we store the default full size here.
_MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
 
 
class OpenAIEmbedding(BaseEmbedding):
    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embedding_model
        self._dims = _MODEL_DIMENSIONS.get(self._model, settings.embedding_dimensions)
        log.info("embedding.provider_init", provider="openai", model=self._model, dims=self._dims)
 
    @property
    def dimensions(self) -> int:
        return self._dims
 
    @property
    def provider_name(self) -> str:
        return f"openai/{self._model}"
 
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=text.strip(),
        )
        return response.data[0].embedding
 
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(
            model=self._model,
            input=[t.strip() for t in texts],
        )
        # OpenAI guarantees order matches input, but sort defensively
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
 