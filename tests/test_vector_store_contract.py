"""Contrato de `VectorStorePort`: todo adaptador precisa passar nesta suíte.

Inclui o isolamento entre clientes: é o teste que prova que um cliente não lê,
não sobrescreve e não apaga os trechos de outro.
"""

from uuid import uuid4

import pytest

from aiplatform.domain.models import Chunk, TenantId
from aiplatform.infrastructure.local import HashingEmbedder, InMemoryVectorStore


def _memory():
    return InMemoryVectorStore()


def _chroma():
    pytest.importorskip("chromadb")
    from aiplatform.infrastructure.chroma_store import ChromaVectorStore

    # Clientes efêmeros do Chroma compartilham estado no processo: prefixo único por teste.
    return ChromaVectorStore(collection_prefix=f"t{uuid4().hex[:12]}")


@pytest.fixture(params=[_memory, _chroma], ids=["memory", "chroma"])
def store(request):
    return request.param()


ACME = TenantId("acme")
GLOBEX = TenantId("globex")
EMB = HashingEmbedder()
CHUNKS = [
    Chunk("pol", 0, "prazo de reembolso de sete dias corridos", "politica.md"),
    Chunk("pol", 1, "reembolso exige nota fiscal original", "politica.md"),
    Chunk("sobre", 0, "empresa fundada em Porto Alegre", "sobre.md"),
]


async def _vector(text):
    [vector] = await EMB.embed([text])
    return vector


async def _fill(store, tenant=ACME, chunks=CHUNKS):
    await store.upsert(tenant, chunks, await EMB.embed([c.text for c in chunks]))


async def test_empty_store_finds_nothing(store):
    assert await store.search(ACME, await _vector("reembolso"), 5) == []


async def test_exact_text_comes_first_with_score_near_one(store):
    await _fill(store)
    [best, *_] = await store.search(ACME, await _vector(CHUNKS[2].text), 3)
    assert best.chunk.id == "sobre#0"
    assert best.score == pytest.approx(1.0, abs=1e-3)


async def test_results_are_sorted_and_respect_top_k(store):
    await _fill(store)
    hits = await store.search(ACME, await _vector("reembolso"), 2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


async def test_upsert_with_same_id_replaces_text(store):
    await _fill(store)
    new = Chunk("sobre", 0, "sede nova em Florianópolis", "sobre.md")
    await _fill(store, chunks=[new])
    hits = await store.search(ACME, await _vector(new.text), 5)
    assert [h.chunk.text for h in hits if h.chunk.id == "sobre#0"] == [new.text]


async def test_delete_document_removes_only_that_document(store):
    await _fill(store)
    await store.delete_document(ACME, "pol")
    hits = await store.search(ACME, await _vector("reembolso prazo empresa"), 5)
    assert {h.chunk.document_id for h in hits} == {"sobre"}


async def test_fields_round_trip(store):
    await _fill(store)
    hits = await store.search(ACME, await _vector(CHUNKS[1].text), 3)
    assert next(h.chunk for h in hits if h.chunk.id == "pol#1") == CHUNKS[1]


async def test_tenant_never_sees_another_tenants_chunks(store):
    await _fill(store, ACME)
    assert await store.search(GLOBEX, await _vector(CHUNKS[0].text), 5) == []


async def test_same_document_id_coexists_across_tenants(store):
    theirs = Chunk("pol", 0, "política da globex sobre férias", "ferias.md")
    await _fill(store, ACME)
    await _fill(store, GLOBEX, [theirs])
    acme_hits = await store.search(ACME, await _vector(CHUNKS[0].text), 1)
    globex_hits = await store.search(GLOBEX, await _vector(theirs.text), 5)
    assert acme_hits[0].chunk == CHUNKS[0]
    assert [h.chunk for h in globex_hits] == [theirs]


async def test_delete_in_one_tenant_keeps_the_other(store):
    await _fill(store, ACME)
    await _fill(store, GLOBEX)
    await store.delete_document(ACME, "pol")
    hits = await store.search(GLOBEX, await _vector("reembolso"), 5)
    assert {h.chunk.document_id for h in hits} >= {"pol"}
