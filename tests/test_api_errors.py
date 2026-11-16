import pytest
from fastapi.testclient import TestClient

from aiplatform.application.ports import ChatModelPort, ProviderUnavailableError
from aiplatform.infrastructure.local import HashingEmbedder, InMemoryEventBus, InMemoryVectorStore
from aiplatform.web.app import create_app


class RaisingModel(ChatModelPort):
    def __init__(self, exc):
        self._exc = exc

    async def complete(self, messages, tools):
        raise self._exc


def _client(exc):
    app = create_app(
        RaisingModel(exc), HashingEmbedder(), InMemoryVectorStore(), InMemoryEventBus()
    )
    client = TestClient(app, raise_server_exceptions=False)
    doc = {"id": "d", "text": "reembolso em sete dias", "source": "p"}
    assert client.post("/documents", json=doc).status_code == 201
    return client


def test_provider_failure_becomes_502_and_is_counted():
    client = _client(ProviderUnavailableError("fora do ar"))
    assert client.post("/ask", json={"question": "reembolso"}).status_code == 502
    assert 'http_requests_total{path="/ask",status="502"} 1.0' in client.get("/metrics").text


def test_unexpected_error_becomes_500_with_request_id_and_is_counted():
    client = _client(RuntimeError("bug"))
    r = client.post("/ask", json={"question": "reembolso"}, headers={"x-request-id": "abc"})
    assert r.status_code == 500
    assert r.json()["request_id"] == "abc" and r.headers["x-request-id"] == "abc"
    assert 'http_requests_total{path="/ask",status="500"} 1.0' in client.get("/metrics").text


@pytest.mark.parametrize("bad", ["x" * 200, "tem espaco", "<script>"])
def test_untrusted_request_id_is_replaced(bad):
    client = _client(RuntimeError("bug"))
    r = client.get("/health", headers={"x-request-id": bad})
    assert r.headers["x-request-id"] != bad and len(r.headers["x-request-id"]) == 12


def test_openapi_describes_the_ask_response():
    schema = _client(RuntimeError("bug")).get("/openapi.json").json()
    ask = schema["paths"]["/ask"]["post"]["responses"]["200"]["content"]["application/json"]
    ref = ask["schema"]["$ref"].rsplit("/", 1)[-1]
    assert {"answer", "grounded", "citations"} <= set(
        schema["components"]["schemas"][ref]["properties"]
    )
