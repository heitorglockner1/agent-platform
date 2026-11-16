"""Adaptadores locais, sem rede nem chave: rodam em CI e na demo.

Servem de referência de contrato para os adaptadores reais (Chroma,
OpenSearch, SQS, Bedrock): a mesma suíte de testes vale para todos.
"""

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any

from aiplatform.application.ports import EmbedderPort, EventPublisherPort, VectorStorePort
from aiplatform.domain.models import Chunk, ScoredChunk

_DIM = 256


class HashingEmbedder(EmbedderPort):
    """Bag-of-words por hashing, normalizado. Determinístico e offline."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    @staticmethod
    def _one(text: str) -> list[float]:
        vec = [0.0] * _DIM
        for tok in re.findall(r"\w{3,}", text.lower()):  # sem palavras funcionais curtas
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % _DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class InMemoryVectorStore(VectorStorePort):
    def __init__(self) -> None:
        self._rows: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        for chunk, vec in zip(chunks, vectors, strict=True):
            self._rows[chunk.id] = (chunk, vec)

    def search(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(c, sum(a * b for a, b in zip(vector, v, strict=True)))
            for c, v in self._rows.values()
        ]
        return sorted(scored, key=lambda s: s.score, reverse=True)[:top_k]

    def delete_document(self, document_id: str) -> None:
        for key in [k for k, (c, _) in self._rows.items() if c.document_id == document_id]:
            del self._rows[key]


class InMemoryEventBus(EventPublisherPort):
    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self.published.append((topic, payload))
