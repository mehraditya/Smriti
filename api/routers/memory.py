from uuid import UUID

import structlog
from fastapi import APIRouter, Query

from api.dependencies import DBSession, Embedding, Graph, Qdrant, Retrieval, State
from api.schemas import (
    AddMemoryRequest,
    AddMemoryResponse,
    GraphResponse,
    MemoryEdgeResponse,
    MemoryNodeResponse,
    QueryMemoryRequest,
    QueryMemoryResponse,
    QueryResultItem,
    RelatedNodeResponse,
)
from core.models import MemoryNode

log = structlog.get_logger()

router = APIRouter(prefix="/memory", tags=["memory"])

@router.post("/add", response_model= AddMemoryResponse, status_code= 201)
async def add_memory(
    body: AddMemoryRequest,
    session: DBSession,
    state_engine: State,
    embedding: Embedding,
    graph: Graph,
    qdrant: Qdrant,
) -> AddMemoryResponse:
    """
    ACCEPT raw text, extract atomic memories, classify relations,
    create/archive nodes based on state, and return.
    """
    from core.embedding import EmbeddingEngine as _Emb
    
    ingestion = state_engine._ingestion
    #1. Atomic memory Extraction from raw input
    input_hash = _Emb.hash_input(body.input)
    extracted = await ingestion.extract(body.input)

    if not extracted:
        log.warning("add_memory.no_memories_extracted", user_id= body.user_id)
        return AddMemoryResponse(nodes_created=[], edges_created=[], nodes_archived=[])
    
    #2. Embed to a single batch call
    texts = [e.content for e in extracted]
    vectors = await embedding.embed_batch(texts)