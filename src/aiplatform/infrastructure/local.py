"""Adaptadores locais, sem rede nem chave: rodam em CI e na demo.

Servem de referência de contrato para os adaptadores reais (Chroma,
OpenSearch, SQS, Bedrock): a mesma suíte de testes vale para todos.
"""

import hashlib
import math
import re
from typing import Any, Dict, List, Sequence, Tuple

from aiplatform.application.ports import EmbedderPort, EventPublisherPort, VectorStorePort
from aiplatform.domain.models import Chunk, ScoredChunk

_DIM = 256


class HashingEmbedder(EmbedderPort):
    """Bag-of-words por hashing, normalizado. Determinístico e offline."""

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._one(t) for t in texts]

    @staticmethod
    def _one(text: str) -> List[float]:
        vec = [0.0] * _DIM
        for tok in re.findall(r"\w{3,}", text.lower()):  # sem palavras funcionais curtas
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % _DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class InMemoryVectorStore(VectorStorePort):
    def __init__(self) -> None:
        self._rows: Dict[str, Tuple[Chunk, List[float]]] = {}

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[List[float]]) -> None:
        for chunk, vec in zip(chunks, vectors):
            self._rows[chunk.id] = (chunk, vec)

    def search(self, vector: List[float], top_k: int) -> List[ScoredChunk]:
        scored = [
            ScoredChunk(c, sum(a * b for a, b in zip(vector, v))) for c, v in self._rows.values()
        ]
        return sorted(scored, key=lambda s: s.score, reverse=True)[:top_k]

    def delete_document(self, document_id: str) -> None:
        for key in [k for k, (c, _) in self._rows.items() if c.document_id == document_id]:
            del self._rows[key]


class InMemoryEventBus(EventPublisherPort):
    def __init__(self) -> None:
        self.published: List[Tuple[str, Dict[str, Any]]] = []

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        self.published.append((topic, payload))
