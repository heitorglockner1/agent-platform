from fastapi.testclient import TestClient

from aiplatform.application.rag import NO_CONTEXT_REPLY, AnswerQuestion, IndexDocument, chunk_text
from aiplatform.domain.models import Document
from aiplatform.infrastructure.fakes import EchoContextModel
from aiplatform.infrastructure.local import HashingEmbedder, InMemoryEventBus, InMemoryVectorStore
from aiplatform.web.app import create_app


def test_chunking_overlaps_and_covers_everything():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 3
    assert chunks[0].split()[-1] in chunks[1].split()  # sobreposição
    assert "w499" in chunks[-1]


def _stack():
    store, bus, emb = InMemoryVectorStore(), InMemoryEventBus(), HashingEmbedder()
    return (
        store,
        bus,
        emb,
        IndexDocument(emb, store, bus),
        AnswerQuestion(EchoContextModel(), emb, store),
    )


async def test_answer_has_citations_from_retrieval():
    _, _, _, index, ask = _stack()
    index(Document("d1", "O prazo de reembolso é de sete dias corridos.", "politica.md"))
    index(Document("d2", "A empresa foi fundada em 1995 em Porto Alegre.", "sobre.md"))
    a = await ask("qual o prazo de reembolso")
    assert a.grounded and a.citations[0].source == "politica.md"


async def test_refuses_instead_of_inventing_when_nothing_relevant():
    _, _, _, index, ask = _stack()
    index(Document("d1", "O prazo de reembolso é de sete dias.", "politica.md"))
    a = await ask("receita de bolo de cenoura")
    assert not a.grounded and a.text == NO_CONTEXT_REPLY


def test_reindex_removes_old_chunks_and_publishes_event():
    store, bus, emb, index, _ = _stack()
    index(Document("d1", "alfa beta gama", "a"))
    index(Document("d1", "totalmente diferente agora", "a"))
    texts = [s.chunk.text for s in store.search(emb.embed(["alfa"])[0], 10)]
    assert texts == ["totalmente diferente agora"]
    assert [t for t, _ in bus.published] == ["document.indexed"] * 2


def test_api_flow_metrics_and_request_id():
    store, bus, emb, _, _ = _stack()
    client = TestClient(create_app(EchoContextModel(), emb, store, bus))
    assert (
        client.post(
            "/documents", json={"id": "d", "text": "reembolso em sete dias", "source": "p"}
        ).status_code
        == 201
    )
    r = client.post("/ask", json={"question": "reembolso"}, headers={"x-request-id": "abc"})
    assert r.json()["grounded"] is True and r.headers["x-request-id"] == "abc"
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert 'http_requests_total{path="/ask",status="200"} 1.0' in client.get("/metrics").text
