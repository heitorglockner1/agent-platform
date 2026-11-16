import pytest

pytest.importorskip("langchain_core")

from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings  # noqa: E402

from aiplatform.application.ports import ProviderUnavailableError  # noqa: E402
from aiplatform.infrastructure.langchain_embeddings import LangChainEmbedder  # noqa: E402


async def test_one_vector_per_text_and_deterministic():
    embedder = LangChainEmbedder(DeterministicFakeEmbedding(size=16))
    first, second = await embedder.embed(["alfa", "beta"])
    assert len(first) == len(second) == 16
    assert await embedder.embed(["alfa"]) == [first]


class BrokenEmbeddings(Embeddings):
    def embed_documents(self, texts):
        raise ConnectionError("sem rede")

    def embed_query(self, text):
        raise ConnectionError("sem rede")


async def test_provider_failure_is_unavailable():
    with pytest.raises(ProviderUnavailableError):
        await LangChainEmbedder(BrokenEmbeddings()).embed(["alfa"])
