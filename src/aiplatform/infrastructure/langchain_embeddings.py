"""Adaptador LangChain para `EmbedderPort` (extra: `pip install .[langchain]`).

Aceita qualquer `Embeddings` do LangChain (OpenAIEmbeddings, BedrockEmbeddings...).
Limitação: a porta tem um único `embed`, então modelos que distinguem consulta de
documento (`embed_query` com instrução própria) são usados sempre no modo documento.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from aiplatform.application.ports import EmbedderPort, ProviderUnavailableError

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


class LangChainEmbedder(EmbedderPort):
    def __init__(self, embeddings: "Embeddings") -> None:
        self._embeddings = embeddings

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            return await self._embeddings.aembed_documents(list(texts))
        except Exception as exc:
            raise ProviderUnavailableError(f"{type(exc).__name__}: {exc}") from exc
