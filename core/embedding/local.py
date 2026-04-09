"""
Local embedding probably will use BAAI/bge/small
Model to be loaded once at first use
"""


import asyncio
import structlog

from config import get_settings
from core.embedding.base import BaseEmbedding

log = structlog.get_logger()


class LocalEmbedding(BaseEmbedding):
    def __init__(self) -> None:
        settings = get_settings()
        self._model_name = settings.local_embedding_model
        self._batch_size = settings.local_embedding_batch_size
        self._device = settings.local_embedding_device

        self._model = None
        self._dims = None

        log.info(
            "embedding.provider_init",
            provider="local",
            model=self._model_name,
            device=self._device,
        )

    def _load_model(self) -> None:
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. Install with: uv add sentence-transformers"
            ) from exc

        log.info(
            "embedding.local_model_loading",
            model=self._model_name,
            device=self._device,
        )

        self._model = SentenceTransformer(self._model_name, device=self._device)

        self._dims = self._model.get_sentence_embedding_dimension()

        log.info(
            "embedding.local_model_ready",
            model=self._model_name,
            dims=self._dims,
        )

    @property
    def dimensions(self) -> int:
        if self._dims is None:
            self._load_model()
        return self._dims

    @property
    def provider_name(self) -> str:
        return f"local/{self._model_name}"

    async def embed(self, text: str) -> list[float]:
        if not text.strip():
            return []

        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._embed_sync,
            text.strip(),
        )

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        cleaned = [t.strip() for t in texts if t.strip()]

        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._embed_batch_sync,
            cleaned,
        )

    def _embed_sync(self, text: str) -> list[float]:
        self._load_model()

        vector = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        self._load_model()

        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]