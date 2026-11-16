"""RAG: indexação (chunking com sobreposição) e resposta com citações.

Decisões:
* `min_score`: abaixo do limiar o sistema recusa em vez de deixar o modelo
  inventar. `Answer.grounded` deixa a UI/chamador saber.
* As citações vêm do código (o que foi recuperado), nunca do texto do modelo.
* Reindexar substitui todos os chunks antigos (sem fantasmas), mas só depois
  de gerar os embeddings novos: se o provedor falhar, a versão antiga fica.
* `Retriever` concentra busca e limiar, usado pelo /ask e pela ferramenta do agente.
"""

from collections.abc import Sequence

from aiplatform.application.ports import (
    ChatModelPort,
    EmbedderPort,
    EventPublisherPort,
    Tracer,
    VectorStorePort,
)
from aiplatform.domain.models import AgentMessage, Answer, Chunk, Citation, Document, ScoredChunk

CONTEXT_HEADER = "CONTEXTO:"
QUESTION_HEADER = "PERGUNTA:"
NO_CONTEXT_REPLY = "Não encontrei isso nos documentos indexados."
GROUNDING_INSTRUCTION = "Responda só com base no CONTEXTO. Se não estiver nele, diga que não sabe."


def _tail(words: list[str], overlap: int) -> list[str]:
    """Maior sufixo de `words` que cabe em `overlap` caracteres."""
    kept = 0
    count = 0
    for word in reversed(words):
        if kept + len(word) + 1 > overlap:
            break
        kept += len(word) + 1
        count += 1
    return words[len(words) - count :]


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Janela por palavras de ~`size` caracteres, repetindo até `overlap` caracteres.

    Como a sobreposição é sempre menor que `size`, todo trecho traz ao menos uma
    palavra nova.
    """
    if size <= 0:
        raise ValueError(f"size deve ser positivo, recebeu {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap deve estar em [0, size), recebeu {overlap} com size={size}")

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    fresh = 0  # palavras ainda não emitidas em nenhum trecho
    for word in text.split():
        current.append(word)
        length += len(word) + 1
        fresh += 1
        if length >= size:
            chunks.append(" ".join(current))
            current = _tail(current, overlap)
            length = sum(len(w) + 1 for w in current)
            fresh = 0
    if fresh:
        chunks.append(" ".join(current))
    return chunks


class IndexDocument:
    """Caso de uso: quebra o documento em trechos, gera embeddings e indexa."""

    def __init__(
        self,
        embedder: EmbedderPort,
        store: VectorStorePort,
        events: EventPublisherPort,
        chunk_size: int = 800,
        overlap: int = 100,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._events = events
        self._chunk_size = chunk_size
        self._overlap = overlap

    async def __call__(self, doc: Document) -> int:
        """Indexa (ou reindexa) o documento e devolve quantos trechos ele gerou."""
        pieces = chunk_text(doc.text, self._chunk_size, self._overlap)
        chunks = [Chunk(doc.id, i, t, doc.source) for i, t in enumerate(pieces)]
        # Embeddings antes de apagar: se o provedor falhar, a versão antiga continua.
        vectors = await self._embedder.embed([c.text for c in chunks]) if chunks else []
        await self._store.delete_document(doc.id)
        if chunks:
            await self._store.upsert(chunks, vectors)
        await self._events.publish(
            "document.indexed", {"document_id": doc.id, "chunks": len(chunks)}
        )
        return len(chunks)


class Retriever:
    """Busca por similaridade com limiar: abaixo de `min_score` o trecho não conta."""

    def __init__(
        self,
        embedder: EmbedderPort,
        store: VectorStorePort,
        top_k: int = 4,
        min_score: float = 0.2,
        tracer: Tracer | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._min_score = min_score
        self._tracer = tracer

    async def __call__(self, query: str) -> list[ScoredChunk]:
        """Trechos relevantes para `query`, do mais para o menos parecido."""
        [vector] = await self._embedder.embed([query])
        found = await self._store.search(vector, self._top_k)
        hits = [h for h in found if h.score >= self._min_score]
        if self._tracer:
            # top_score antes do limiar: é o número que serve para calibrá-lo.
            top = round(found[0].score, 4) if found else None
            self._tracer.event("rag.retrieval", hits=len(hits), top_score=top)
        return hits


class AnswerQuestion:
    """Caso de uso: pipeline de RAG com recusa abaixo do limiar e citações do código."""

    def __init__(self, model: ChatModelPort, retrieve: Retriever) -> None:
        self._model = model
        self._retrieve = retrieve

    async def __call__(self, question: str) -> Answer:
        hits = await self._retrieve(question)
        if not hits:
            return Answer(NO_CONTEXT_REPLY, (), grounded=False)  # o modelo nem é chamado

        reply = await self._model.complete(grounded_prompt(question, hits), [])
        citations = tuple(Citation(h.chunk.id, h.chunk.source, round(h.score, 4)) for h in hits)
        return Answer(reply.content, citations, grounded=True)


def grounded_prompt(question: str, hits: Sequence[ScoredChunk]) -> list[AgentMessage]:
    """Prompt que restringe o modelo aos trechos recuperados, cada um com seu id."""
    context = "\n\n".join(f"[{h.chunk.id}] {h.chunk.text}" for h in hits)
    return [
        AgentMessage("system", GROUNDING_INSTRUCTION),
        AgentMessage("user", f"{CONTEXT_HEADER}\n{context}\n\n{QUESTION_HEADER} {question}"),
    ]
