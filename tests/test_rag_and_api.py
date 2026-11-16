import pytest
from fastapi.testclient import TestClient

from aiplatform.application.rag import (
    NO_CONTEXT_REPLY,
    AnswerQuestion,
    IndexDocument,
    Retriever,
    chunk_text,
)
from aiplatform.domain.models import Document
from aiplatform.infrastructure.demo import DemoChatModel
from aiplatform.infrastructure.local import HashingEmbedder, InMemoryEventBus, InMemoryVectorStore
from aiplatform.web.app import create_app


def test_chunking_overlaps_and_covers_everything():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 3
    assert chunks[0].split()[-1] in chunks[1].split()  # sobreposição
    assert "w499" in chunks[-1]


def test_chunking_does_not_emit_a_chunk_made_only_of_overlap():
    assert len(chunk_text(" ".join(["abcd"] * 40), size=200, overlap=40)) == 1


def test_every_chunk_adds_new_words_and_rebuilds_the_text():
    words = [f"w{i}" for i in range(500)]
    chunks = [c.split() for c in chunk_text(" ".join(words), size=200, overlap=40)]
    rebuilt = list(chunks[0])
    for chunk in chunks[1:]:
        start = rebuilt.index(chunk[0]) if chunk[0] in rebuilt else len(rebuilt)
        assert len(rebuilt) - start < len(chunk)  # ao menos uma palavra nova
        rebuilt = rebuilt[:start] + chunk
    assert rebuilt == words


def test_zero_overlap_repeats_nothing():
    chunks = [set(c.split()) for c in chunk_text(" ".join(f"w{i}" for i in range(300)), 100, 0)]
    assert all(not (a & b) for a, b in zip(chunks, chunks[1:], strict=False))


@pytest.mark.parametrize(("size", "overlap"), [(100, 100), (100, 150), (0, 0), (-5, 0), (100, -1)])
def test_invalid_chunk_parameters_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("texto qualquer", size, overlap)


def test_empty_text_has_no_chunks():
    assert chunk_text("   ") == []


POLICY = "O prazo de reembolso é de sete dias corridos."


class Stack:
    """Adaptadores locais e os dois casos de uso de RAG ligados a eles."""

    def __init__(self):
        self.store = InMemoryVectorStore()
        self.bus = InMemoryEventBus()
        self.embedder = HashingEmbedder()
        self.index = IndexDocument(self.embedder, self.store, self.bus)
        self.ask = AnswerQuestion(DemoChatModel(), Retriever(self.embedder, self.store))

    async def texts_near(self, query):
        [vector] = await self.embedder.embed([query])
        return [hit.chunk.text for hit in await self.store.search(vector, 10)]


async def test_answer_has_citations_from_retrieval():
    stack = Stack()
    await stack.index(Document("d1", POLICY, "politica.md"))
    await stack.index(Document("d2", "A empresa foi fundada em 1995 em Porto Alegre.", "sobre.md"))
    a = await stack.ask("qual o prazo de reembolso")
    assert a.grounded and a.citations[0].source == "politica.md"


async def test_demo_model_answers_with_the_retrieved_text():
    stack = Stack()
    await stack.index(Document("pol", POLICY, "politica.md"))
    a = await stack.ask("qual o prazo de reembolso?")
    assert a.text == POLICY


async def test_refuses_instead_of_inventing_when_nothing_relevant():
    stack = Stack()
    await stack.index(Document("d1", "O prazo de reembolso é de sete dias.", "politica.md"))
    a = await stack.ask("receita de bolo de cenoura")
    assert not a.grounded and a.text == NO_CONTEXT_REPLY


async def test_reindex_removes_old_chunks_and_publishes_event():
    stack = Stack()
    await stack.index(Document("d1", "alfa beta gama", "a"))
    await stack.index(Document("d1", "totalmente diferente agora", "a"))
    texts = await stack.texts_near("alfa")
    assert texts == ["totalmente diferente agora"]
    assert [t for t, _ in stack.bus.published] == ["document.indexed"] * 2


class FailingEmbedder(HashingEmbedder):
    async def embed(self, texts):
        raise RuntimeError("provedor fora do ar")


async def test_reindex_keeps_old_version_when_embedding_fails():
    stack = Stack()
    await stack.index(Document("d1", "alfa beta gama", "a"))
    with pytest.raises(RuntimeError):
        await IndexDocument(FailingEmbedder(), stack.store, stack.bus)(
            Document("d1", "versão nova", "a")
        )
    texts = await stack.texts_near("alfa")
    assert texts == ["alfa beta gama"]


def test_api_flow_metrics_and_request_id():
    stack = Stack()
    client = TestClient(create_app(DemoChatModel(), stack.embedder, stack.store, stack.bus))
    assert (
        client.post(
            "/documents", json={"id": "d", "text": "reembolso em sete dias", "source": "p"}
        ).status_code
        == 201
    )
    r = client.post("/ask", json={"question": "reembolso"}, headers={"x-request-id": "abc"})
    assert r.json()["grounded"] is True and r.headers["x-request-id"] == "abc"
    assert r.json()["answer"] == "reembolso em sete dias"
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert 'http_requests_total{path="/ask",status="200"} 1.0' in client.get("/metrics").text
