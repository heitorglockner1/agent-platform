"""RAG: indexação (chunking com sobreposição) e resposta com citações.

Decisões:
* `min_score`: abaixo do limiar o sistema recusa em vez de deixar o modelo
  inventar. `Answer.grounded` deixa a UI/chamador saber.
* As citações vêm do código (o que foi recuperado), nunca do texto do modelo.
* Re-indexar um documento apaga os chunks antigos antes (sem fantasmas).
"""

from typing import List, Sequence

from aiplatform.application.ports import (
    ChatModelPort,
    EmbedderPort,
    EventPublisherPort,
    VectorStorePort,
)
from aiplatform.domain.models import AgentMessage, Answer, Chunk, Citation, Document

NO_CONTEXT_REPLY = "Não encontrei isso nos documentos indexados."


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> List[str]:
    """Janela por palavras, ~`size` caracteres, com `overlap` de sobreposição."""
    chunks: List[str] = []
    current: List[str] = []
    length = 0
    for word in text.split():
        current.append(word)
        length += len(word) + 1
        if length >= size:
            chunks.append(" ".join(current))
            keep: List[str] = []
            back = 0
            for w in reversed(current):
                back += len(w) + 1
                keep.insert(0, w)
                if back >= overlap:
                    break
            current, length = keep, sum(len(w) + 1 for w in keep)
    if current and (not chunks or " ".join(current) != chunks[-1]):
        chunks.append(" ".join(current))
    return chunks


class IndexDocument:
    def __init__(
        self,
        embedder: EmbedderPort,
        store: VectorStorePort,
        events: EventPublisherPort,
        chunk_size: int = 800,
        overlap: int = 100,
    ) -> None:
        self._embedder, self._store, self._events = embedder, store, events
        self._size, self._overlap = chunk_size, overlap

    def __call__(self, doc: Document) -> int:
        pieces = chunk_text(doc.text, self._size, self._overlap)
        chunks = [Chunk(doc.id, i, t, doc.source) for i, t in enumerate(pieces)]
        self._store.delete_document(doc.id)
        if chunks:
            self._store.upsert(chunks, self._embedder.embed([c.text for c in chunks]))
        self._events.publish("document.indexed", {"document_id": doc.id, "chunks": len(chunks)})
        return len(chunks)


class AnswerQuestion:
    def __init__(
        self,
        model: ChatModelPort,
        embedder: EmbedderPort,
        store: VectorStorePort,
        top_k: int = 4,
        min_score: float = 0.2,
    ) -> None:
        self._model, self._embedder, self._store = model, embedder, store
        self._top_k, self._min_score = top_k, min_score

    async def __call__(self, question: str) -> Answer:
        hits = self._store.search(self._embedder.embed([question])[0], self._top_k)
        hits = [h for h in hits if h.score >= self._min_score]
        if not hits:
            return Answer(NO_CONTEXT_REPLY, [], grounded=False)

        context = "\n\n".join(f"[{h.chunk.id}] {h.chunk.text}" for h in hits)
        messages: Sequence[AgentMessage] = [
            AgentMessage(
                role="system",
                content="Responda só com base no CONTEXTO. Se não estiver nele, diga que não sabe.",
            ),
            AgentMessage(role="user", content=f"CONTEXTO:\n{context}\n\nPERGUNTA: {question}"),
        ]
        reply = await self._model.complete(messages, [])
        citations = [Citation(h.chunk.id, h.chunk.source, round(h.score, 4)) for h in hits]
        return Answer(reply.content, citations, grounded=True)
