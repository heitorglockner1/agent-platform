"""Adaptadores locais, sem rede nem chave: rodam em CI e na demo.

Servem de referência de contrato para os adaptadores reais: a suíte de
`tests/test_vector_store_contract.py` roda contra este store e contra o Chroma.
"""

import hashlib
import math
import re
from collections import deque
from collections.abc import Sequence
from typing import Any

from aiplatform.application.ports import EmbedderPort, EventPublisherPort, VectorStorePort
from aiplatform.domain.models import Chunk, ScoredChunk

_DIM = 256


class HashingEmbedder(EmbedderPort):
    """Bag-of-words por hashing, normalizado. Determinístico e offline."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    @staticmethod
    def _one(text: str) -> list[float]:
        vector = [0.0] * _DIM
        # Tokens de 3+ letras: "de" e "o" faziam perguntas sem relação casarem (LEARNINGS).
        for token in re.findall(r"\w{3,}", text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % _DIM
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class InMemoryVectorStore(VectorStorePort):
    """Store em memória. Perde os dados ao reiniciar.

    Métodos async sem await: cada operação roda inteira no event loop, então não
    há corrida entre requisições e não é preciso lock.
    """

    def __init__(self) -> None:
        self._rows: dict[str, tuple[Chunk, list[float]]] = {}

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        for chunk, vec in zip(chunks, vectors, strict=True):
            self._rows[chunk.id] = (chunk, vec)

    async def search(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        scored = [ScoredChunk(chunk, _dot(vector, stored)) for chunk, stored in self._rows.values()]
        return sorted(scored, key=lambda s: s.score, reverse=True)[:top_k]

    async def delete_document(self, document_id: str) -> None:
        doomed = [key for key, (chunk, _) in self._rows.items() if chunk.document_id == document_id]
        for key in doomed:
            del self._rows[key]


def _dot(a: list[float], b: list[float]) -> float:
    """Com vetores normalizados, o produto escalar é a similaridade de cosseno."""
    return sum(x * y for x, y in zip(a, b, strict=True))


class InMemoryEventBus(EventPublisherPort):
    """Guarda os últimos eventos publicados. Não há consumidor: serve para teste e demo."""

    def __init__(self) -> None:
        self.published: deque[tuple[str, dict[str, Any]]] = deque(maxlen=1000)

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self.published.append((topic, payload))
