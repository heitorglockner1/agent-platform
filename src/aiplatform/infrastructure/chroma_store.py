"""Adaptador ChromaDB (extra opcional: `pip install .[chroma]`).

Implementa a mesma `VectorStorePort` do store em memória. Trocar por
OpenSearch, Weaviate ou Pinecone é escrever outro arquivo deste tamanho.
"""

from typing import List, Optional, Sequence

from aiplatform.application.ports import VectorStorePort
from aiplatform.domain.models import Chunk, ScoredChunk


class ChromaVectorStore(VectorStorePort):
    def __init__(self, collection: str = "documents", path: Optional[str] = None) -> None:
        import chromadb  # import tardio: dependência opcional

        client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        self._col = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[List[float]]) -> None:
        self._col.upsert(
            ids=[c.id for c in chunks],
            embeddings=[list(v) for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[{"document_id": c.document_id, "index": c.index, "source": c.source} for c in chunks],
        )

    def search(self, vector: List[float], top_k: int) -> List[ScoredChunk]:
        res = self._col.query(query_embeddings=[vector], n_results=top_k)
        out: List[ScoredChunk] = []
        for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            chunk = Chunk(meta["document_id"], int(meta["index"]), text, meta.get("source", ""))
            out.append(ScoredChunk(chunk, 1.0 - float(dist)))  # cosseno: distância -> similaridade
        return out

    def delete_document(self, document_id: str) -> None:
        self._col.delete(where={"document_id": document_id})
