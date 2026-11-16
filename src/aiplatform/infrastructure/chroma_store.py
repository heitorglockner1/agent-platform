"""Adaptador ChromaDB (extra opcional: `pip install .[chroma]`).

Implementa a mesma `VectorStorePort` do store em memória. Trocar por
OpenSearch, Weaviate ou Pinecone é escrever outro arquivo deste tamanho.
"""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from aiplatform.application.ports import VectorStorePort
from aiplatform.domain.models import Chunk, ScoredChunk


class ChromaVectorStore(VectorStorePort):
    """O cliente do Chroma é síncrono: cada chamada roda em thread para não travar o loop."""

    def __init__(self, collection: str = "documents", path: str | None = None) -> None:
        import chromadb  # import tardio: dependência opcional

        client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        self._col = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        await asyncio.to_thread(
            self._col.upsert,
            ids=[c.id for c in chunks],
            embeddings=[list(v) for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[
                {"document_id": c.document_id, "index": c.index, "source": c.source} for c in chunks
            ],
        )

    async def search(self, vector: list[float], top_k: int) -> list[ScoredChunk]:
        res = await asyncio.to_thread(self._col.query, query_embeddings=[vector], n_results=top_k)
        # Uma consulta só: o Chroma devolve listas de listas, a primeira é a nossa.
        rows = zip(res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True)
        return [_scored(text, meta, distance) for text, meta, distance in rows]

    async def delete_document(self, document_id: str) -> None:
        await asyncio.to_thread(self._col.delete, where={"document_id": document_id})


def _scored(text: str, meta: Mapping[str, Any], distance: float) -> ScoredChunk:
    chunk = Chunk(str(meta["document_id"]), int(meta["index"]), text, str(meta.get("source", "")))
    return ScoredChunk(chunk, 1.0 - float(distance))  # distância de cosseno -> similaridade
