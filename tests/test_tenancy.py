"""Isolamento entre clientes pela API: chave, dados e métricas."""

import pytest
from fastapi.testclient import TestClient

from aiplatform.application.rag import NO_CONTEXT_REPLY
from aiplatform.domain.models import TenantId
from aiplatform.infrastructure.demo import DemoChatModel
from aiplatform.infrastructure.local import HashingEmbedder, InMemoryEventBus, InMemoryVectorStore
from aiplatform.settings import Settings
from aiplatform.web.app import create_app

POLICY = "O prazo de reembolso é de sete dias corridos."
ACME = {"authorization": "Bearer chave-acme"}
GLOBEX = {"authorization": "Bearer chave-globex"}


@pytest.fixture
def client():
    keys = {"chave-acme": TenantId("acme"), "chave-globex": TenantId("globex")}
    app = create_app(
        DemoChatModel(), HashingEmbedder(), InMemoryVectorStore(), InMemoryEventBus(), api_keys=keys
    )
    client = TestClient(app)
    doc = {"id": "pol", "text": POLICY, "source": "politica.md"}
    assert client.post("/documents", json=doc, headers=ACME).status_code == 201
    return client


@pytest.mark.parametrize(
    "headers", [{}, {"authorization": "Bearer errada"}, {"authorization": "chave-acme"}]
)
def test_missing_or_wrong_key_is_401(client, headers):
    r = client.post("/ask", json={"question": "prazo de reembolso"}, headers=headers)
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"


def test_probes_stay_open(client):
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_owner_finds_its_document(client):
    r = client.post("/ask", json={"question": "qual o prazo de reembolso?"}, headers=ACME)
    assert r.json()["grounded"] is True and r.json()["answer"] == POLICY


def test_other_tenant_cannot_read_it_through_ask_or_agent(client):
    ask = client.post("/ask", json={"question": "qual o prazo de reembolso?"}, headers=GLOBEX)
    assert ask.json()["grounded"] is False
    agent = client.post("/agent", json={"task": "qual o prazo de reembolso?"}, headers=GLOBEX)
    assert agent.json()["answer"].startswith(NO_CONTEXT_REPLY)


def test_same_document_id_does_not_overwrite_another_tenant(client):
    other = {"id": "pol", "text": "Férias de trinta dias por ano.", "source": "ferias.md"}
    assert client.post("/documents", json=other, headers=GLOBEX).status_code == 201
    r = client.post("/ask", json={"question": "qual o prazo de reembolso?"}, headers=ACME)
    assert r.json()["answer"] == POLICY


def test_agent_cost_is_reported_per_tenant(client):
    client.post("/agent", json={"task": "qual o prazo de reembolso?"}, headers=ACME)
    metrics = client.get("/metrics").text
    assert 'agent_runs_total{stopped_by="concluded",tenant="acme"} 1.0' in metrics
    assert 'agent_tokens_total{tenant="acme"}' in metrics


def test_api_keys_setting_maps_each_key_to_its_tenant():
    s = Settings.from_env({"API_KEYS": "k1:acme, k2:globex"})
    assert dict(s.api_keys) == {"k1": TenantId("acme"), "k2": TenantId("globex")}


@pytest.mark.parametrize("raw", ["segredo-sem-cliente", ":acme", "segredo:Cliente Inválido"])
def test_invalid_api_keys_fail_without_echoing_the_secret(raw):
    with pytest.raises(ValueError, match="API_KEYS") as err:
        Settings.from_env({"API_KEYS": raw})
    assert "segredo" not in str(err.value)


@pytest.mark.parametrize("bad", ["", "ab", "Acme", "com espaço", "-acme", "a" * 64])
def test_tenant_id_rejects_what_cannot_be_a_collection_name(bad):
    with pytest.raises(ValueError):
        TenantId(bad)
