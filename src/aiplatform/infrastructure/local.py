"""Adaptadores locais, sem rede nem chave: rodam em CI e na demo.

Servem de referência de contrato para os adaptadores reais: a suíte de
`tests/test_vector_store_contract.py` roda contra este store e contra o Chroma.
"""

import hashlib
import math
import re
from collections import defaultdict, deque
from collections.abc import Sequence
from typing import Any

from aiplatform.application.ports import EmbedderPort, EventPublisherPort, VectorStorePort
from aiplatform.domain.models import Chunk, ScoredChunk, TenantId

_DIM = 256
_Row = tuple[Chunk, list[float]]


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
    """Store em memória, com um dicionário separado por cliente. Perde os dados ao reiniciar.

    Métodos async sem await: cada operação roda inteira no event loop, então não
    há corrida entre requisições e não é preciso lock.
    """

    def __init__(self) -> None:
        self._rows: defaultdict[TenantId, dict[str, _Row]] = defaultdict(dict)

    async def upsert(
        self, tenant: TenantId, chunks: Sequence[Chunk], vectors: Sequence[list[float]]
    ) -> None:
        rows = self._rows[tenant]
        for chunk, vector in zip(chunks, vectors, strict=True):
            rows[chunk.id] = (chunk, vector)

    async def search(self, tenant: TenantId, vector: list[float], top_k: int) -> list[ScoredChunk]:
        rows = self._rows.get(tenant, {})  # get: buscar não cria cliente vazio
        scored = [ScoredChunk(chunk, _dot(vector, stored)) for chunk, stored in rows.values()]
        return sorted(scored, key=lambda s: s.score, reverse=True)[:top_k]

    async def delete_document(self, tenant: TenantId, document_id: str) -> None:
        rows = self._rows.get(tenant, {})
        doomed = [key for key, (chunk, _) in rows.items() if chunk.document_id == document_id]
        for key in doomed:
            del rows[key]


def _dot(a: list[float], b: list[float]) -> float:
    """Com vetores normalizados, o produto escalar é a similaridade de cosseno."""
    return sum(x * y for x, y in zip(a, b, strict=True))


class InMemoryEventBus(EventPublisherPort):
    """Guarda os últimos eventos publicados. Não há consumidor: serve para teste e demo."""

    def __init__(self) -> None:
        self.published: deque[tuple[str, dict[str, Any]]] = deque(maxlen=1000)

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self.published.append((topic, payload))
