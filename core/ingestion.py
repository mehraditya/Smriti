"""
Memory ingestion and relation classification
Depends on BaseLLM 
The concrete provider is injected at construction time via factory
"""

import json

import structlog

from config import get_settings
from core.llm.base import BaseLLM
from core.models import ExtractedMemory, MemoryType

log = structlog.get_logger()
settings = get_settings()

_EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction engine.
Given a text input, extract all atomic facts, preferences, events, and named entities.
 
Rules:
- Each memory must be self-contained and declarative.
- One subject per memory. Split compound statements.
- Avoid duplicates within the same extraction.
- Use third-person phrasing: "User lives in Delhi" not "I live in Delhi".
- Assign a memory_type: fact | preference | event | entity
- Assign a confidence between 0.0 and 1.0.
 
Return ONLY a JSON array. No preamble, no markdown, no explanation.
Format: [{"content": "...", "memory_type": "fact", "confidence": 0.95}]
"""

_CLASSIFICATION_SYSTEM_PROMPT = """You are a memory relation classifier.
Given two memory statements, classify their relationship.
 
Labels:
- updates: the new statement replaces, contradicts, or supersedes the old one
- extends: the new statement adds non-conflicting detail to the old one
- unrelated: different subjects or topics entirely
 
Return ONLY the label as a single word: updates | extends | unrelated
No explanation, no punctuation.
"""
 
class IngestionEngine:
    # Extract atomic memories from raw text and classifies relations between them.

    def __init__(self, llm: BaseLLM) ->None:
        self._llm = llm

    async def extract(self, raw_input: str) ->list[ExtractedMemory]:
        """
        Send raw text to the LLM and parse out atomic memories.
        Returns an empty list on parse failure (with a warning log) -- never raises.
        """
        raw = await self._llm.complete(
            system
        )
        