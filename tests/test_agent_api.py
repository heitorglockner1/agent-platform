from fastapi.testclient import TestClient

from aiplatform.application.rag import NO_CONTEXT_REPLY
from aiplatform.infrastructure.demo import DemoChatModel
from aiplatform.infrastructure.local import HashingEmbedder, InMemoryEventBus, InMemoryVectorStore
from aiplatform.web.app import create_app
from tests.doubles import RecordingTracer

POLICY = "O prazo de reembolso é de sete dias corridos."


def _client(tracer=None, index=True):
    app = create_app(
        DemoChatModel(), HashingEmbedder(), InMemoryVectorStore(), InMemoryEventBus(), tracer=tracer
    )
    client = TestClient(app)
    if index:
        doc = {"id": "pol", "text": POLICY, "source": "politica.md"}
        assert client.post("/documents", json=doc).status_code == 201
    return client


def test_agent_searches_then_answers_with_the_document():
    client = _client()
    body = client.post("/agent", json={"task": "qual o prazo de reembolso?"}).json()
    assert body["answer"].startswith(POLICY)
    assert "[executado de fato: search_documents]" in body["answer"]
    assert body["stopped_by"] == "concluded" and body["iterations"] == 2
    assert 'agent_runs_total{stopped_by="concluded"} 1.0' in client.get("/metrics").text


def test_agent_says_it_found_nothing_without_documents():
    body = _client(index=False).post("/agent", json={"task": "qual o prazo?"}).json()
    assert body["answer"].startswith(NO_CONTEXT_REPLY)


def test_agent_turns_and_retrieval_are_traced():
    tracer = RecordingTracer()
    _client(tracer).post("/agent", json={"task": "qual o prazo de reembolso?"})
    names = [name for name, _ in tracer.events]
    assert names.count("agent.turn") == 2 and "rag.retrieval" in names


def test_empty_task_is_rejected():
    assert _client(index=False).post("/agent", json={"task": ""}).status_code == 422
