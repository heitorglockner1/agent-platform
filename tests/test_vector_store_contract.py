"""Contrato de `VectorStorePort`: todo adaptador precisa passar nesta suíte."""

from uuid import uuid4

import pytest

from aiplatform.domain.models import Chunk
from aiplatform.infrastructure.local import HashingEmbedder, InMemoryVectorStore


def _memory():
    return InMemoryVectorStore()


def _chroma():
    pytest.importorskip("chromadb")
    from aiplatform.infrastructure.chroma_store import ChromaVectorStore

    # Clientes efêmeros do Chroma compartilham estado no processo: coleção única por teste.
    return ChromaVectorStore(collection=f"t-{uuid4().hex}")


@pytest.fixture(params=[_memory, _chroma], ids=["memory", "chroma"])
def store(request):
    return request.param()


EMB = HashingEmbedder()
CHUNKS = [
    Chunk("pol", 0, "prazo de reembolso de sete dias corridos", "politica.md"),
    Chunk("pol", 1, "reembolso exige nota fiscal original", "politica.md"),
    Chunk("sobre", 0, "empresa fundada em Porto Alegre", "sobre.md"),
]


async def _vector(text):
    [vector] = await EMB.embed([text])
    return vector


async def _fill(store):
    await store.upsert(CHUNKS, await EMB.embed([c.text for c in CHUNKS]))


async def test_empty_store_finds_nothing(store):
    assert await store.search(await _vector("reembolso"), 5) == []


async def test_exact_text_comes_first_with_score_near_one(store):
    await _fill(store)
    [best, *_] = await store.search(await _vector(CHUNKS[2].text), 3)
    assert best.chunk.id == "sobre#0"
    assert best.score == pytest.approx(1.0, abs=1e-3)


async def test_results_are_sorted_and_respect_top_k(store):
    await _fill(store)
    hits = await store.search(await _vector("reembolso"), 2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


async def test_upsert_with_same_id_replaces_text(store):
    await _fill(store)
    new = Chunk("sobre", 0, "sede nova em Florianópolis", "sobre.md")
    await store.upsert([new], await EMB.embed([new.text]))
    hits = await store.search(await _vector(new.text), 5)
    assert [h.chunk.text for h in hits if h.chunk.id == "sobre#0"] == [new.text]


async def test_delete_document_removes_only_that_document(store):
    await _fill(store)
    await store.delete_document("pol")
    hits = await store.search(await _vector("reembolso prazo empresa"), 5)
    assert {h.chunk.document_id for h in hits} == {"sobre"}


async def test_fields_round_trip(store):
    await _fill(store)
    hits = await store.search(await _vector(CHUNKS[1].text), 3)
    assert next(h.chunk for h in hits if h.chunk.id == "pol#1") == CHUNKS[1]
