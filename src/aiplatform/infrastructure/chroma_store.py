"""Adaptador ChromaDB (extra opcional: `pip install .[chroma]`).

Implementa a mesma `VectorStorePort` do store em memória. Trocar por
OpenSearch, Weaviate ou Pinecone é escrever outro arquivo deste tamanho.

Isolamento: uma coleção por cliente (`<prefixo>-<cliente>`), em vez de uma
coleção única filtrada por metadado. A separação fica física: um filtro
esquecido não tem como vazar trechos de outro cliente.
"""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from aiplatform.application.ports import VectorStorePort
from aiplatform.domain.models import Chunk, ScoredChunk, TenantId


class ChromaVectorStore(VectorStorePort):
    """O cliente do Chroma é síncrono: cada chamada roda em thread para não travar o loop."""

    def __init__(self, collection_prefix: str = "documents", path: str | None = None) -> None:
        import chromadb  # import tardio: dependência opcional

        self._client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        self._prefix = collection_prefix
        self._collections: dict[TenantId, Any] = {}

    async def upsert(
        self, tenant: TenantId, chunks: Sequence[Chunk], vectors: Sequence[list[float]]
    ) -> None:
        collection = await self._collection(tenant)
        await asyncio.to_thread(
            collection.upsert,
            ids=[c.id for c in chunks],
            embeddings=[list(v) for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[
                {"document_id": c.document_id, "index": c.index, "source": c.source} for c in chunks
            ],
        )

    async def search(self, tenant: TenantId, vector: list[float], top_k: int) -> list[ScoredChunk]:
        collection = await self._collection(tenant)
        res = await asyncio.to_thread(collection.query, query_embeddings=[vector], n_results=top_k)
        # Uma consulta só: o Chroma devolve listas de listas, a primeira é a nossa.
        rows = zip(res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True)
        return [_scored(text, meta, distance) for text, meta, distance in rows]

    async def delete_document(self, tenant: TenantId, document_id: str) -> None:
        collection = await self._collection(tenant)
        await asyncio.to_thread(collection.delete, where={"document_id": document_id})

    async def _collection(self, tenant: TenantId) -> Any:
        """Coleção do cliente, criada na primeira vez que ele é visto."""
        if tenant not in self._collections:
            self._collections[tenant] = await asyncio.to_thread(
                self._client.get_or_create_collection,
                f"{self._prefix}-{tenant}",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[tenant]


def _scored(text: str, meta: Mapping[str, Any], distance: float) -> ScoredChunk:
    chunk = Chunk(str(meta["document_id"]), int(meta["index"]), text, str(meta.get("source", "")))
    return ScoredChunk(chunk, 1.0 - float(distance))  # distância de cosseno -> similaridade
