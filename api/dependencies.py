from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from core.embedding.base import BaseEmbedding
from core.embedding.factory import get_embedding_engine as _build_embedding
from core.graph import GraphEngine
from core.ingestion import IngestionEngine
from core.llm.base import BaseLLM
from core.llm.factory import get_llm_engine as _build_llm
from core.retrieval import RetrievalEngine
from core.state import StateEngine
from storage.postgres_client import get_db_session
from storage.qdrant_client import QdrantStore


"""Singletons 
Instantiated once per process. lru_cache on a zero-arg function is the
lightest way to do this without a global variable.
"""

@lru_cache
def get_qdrant() -> QdrantStore:
    return QdrantStore()


@lru_cache
def get_embedding_engine() -> BaseEmbedding:
    return _build_embedding()


@lru_cache
def get_llm_engine() -> BaseLLM:
    return _build_llm()


@lru_cache
def get_graph_engine() -> GraphEngine:
    return GraphEngine()


@lru_cache
def get_ingestion_engine(
    llm: BaseLLM = None,
) -> IngestionEngine:
    # Can't use Depends inside lru_cache, so we call get_llm_engine() directly.
    # This is fine because get_llm_engine() is itself cached.
    return IngestionEngine(llm=get_llm_engine())


#Composed engines (not cached — cheap to construct each request)

def get_state_engine(
    ingestion: Annotated[IngestionEngine, Depends(get_ingestion_engine)],
    embedding: Annotated[BaseEmbedding, Depends(get_embedding_engine)],
    graph: Annotated[GraphEngine, Depends(get_graph_engine)],
    qdrant: Annotated[QdrantStore, Depends(get_qdrant)],
) -> StateEngine:
    return StateEngine(ingestion=ingestion, embedding=embedding, graph=graph, qdrant=qdrant)


def get_retrieval_engine(
    embedding: Annotated[BaseEmbedding, Depends(get_embedding_engine)],
    graph: Annotated[GraphEngine, Depends(get_graph_engine)],
    qdrant: Annotated[QdrantStore, Depends(get_qdrant)],
) -> RetrievalEngine:
    return RetrievalEngine(embedding=embedding, graph=graph, qdrant=qdrant)


#Type aliases for endpoint signatures 

DbSession = Annotated[AsyncSession, Depends(get_db_session)]
Qdrant = Annotated[QdrantStore, Depends(get_qdrant)]
Embedding = Annotated[BaseEmbedding, Depends(get_embedding_engine)]
LLM = Annotated[BaseLLM, Depends(get_llm_engine)]
Ingestion = Annotated[IngestionEngine, Depends(get_ingestion_engine)]
Graph = Annotated[GraphEngine, Depends(get_graph_engine)]
State = Annotated[StateEngine, Depends(get_state_engine)]
Retrieval = Annotated[RetrievalEngine, Depends(get_retrieval_engine)]
