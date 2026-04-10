from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_name: str = "memory-system"
    debug: bool = False
    log_level: str = "INFO"

    # Postgres
    postgres_dsn: str = "postgresql+asyncpg://memory:memory@localhost:5432/memory_db"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "memories"
    qdrant_api_key: str | None = None

    # ── Provider selection ────────────────────────────────────────────────────
    # "openai" | "local"
    embedding_provider: str = "openai"
    # "openai" | "local"
    llm_provider: str = "openai"

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    # Override to point at Azure, Together.ai, Groq, etc.
    openai_base_url: str | None = None

    # ── Local embedding (sentence-transformers) ───────────────────────────────
    local_embedding_model: str = "all-MiniLM-L6-v2"
    local_embedding_device: str = "cpu"   # "cpu" | "cuda" | "mps"
    local_embedding_batch_size: int = 64

    # ── Local LLM (Ollama) ────────────────────────────────────────────────────
    local_llm_model: str = "llama3.2"
    local_llm_base_url: str = "http://localhost:11434/v1"

    # ── Shared embedding config ───────────────────────────────────────────────
    # Only used when the model is not in the known-dimensions table.
    # Qdrant collection is created with this size — must match actual output.
    embedding_dimensions: int = 1536

    # ── Retrieval tuning ──────────────────────────────────────────────────────
    retrieval_vector_weight: float = 0.6
    retrieval_graph_weight: float = 0.3
    retrieval_confidence_weight: float = 0.1
    retrieval_similarity_threshold: float = 0.75
    retrieval_default_top_k: int = 5
    retrieval_graph_depth: int = 2

    # ── Ingestion ─────────────────────────────────────────────────────────────
    extraction_temperature: float = 0.1
    classification_temperature: float = 0.0


@lru_cache
def get_settings() -> Settings:
    return Settings()