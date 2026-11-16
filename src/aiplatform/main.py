"""Entrypoint: `uvicorn aiplatform.main:app`. Sem chave, roda com adaptadores locais."""

from fastapi import FastAPI

from aiplatform.application.ports import ChatModelPort, EmbedderPort, VectorStorePort
from aiplatform.infrastructure.demo import EchoContextModel
from aiplatform.infrastructure.local import HashingEmbedder, InMemoryEventBus, InMemoryVectorStore
from aiplatform.observability import configure_logging
from aiplatform.settings import Settings
from aiplatform.web.app import create_app


def _missing_extra(setting: str, extra: str) -> RuntimeError:
    return RuntimeError(f'{setting} exige o extra {extra}: pip install ".[{extra}]"')


def _store(settings: Settings) -> VectorStorePort:
    if settings.vector_store == "chroma":
        try:
            from aiplatform.infrastructure.chroma_store import ChromaVectorStore

            return ChromaVectorStore(path=settings.chroma_path)
        except ImportError as exc:
            raise _missing_extra("VECTOR_STORE=chroma", "chroma") from exc
    return InMemoryVectorStore()


def _model(settings: Settings) -> ChatModelPort:
    if settings.llm == "openai":
        try:
            from langchain_openai import ChatOpenAI

            from aiplatform.infrastructure.langchain_chat import LangChainChatModel
        except ImportError as exc:
            raise _missing_extra("OPENAI_API_KEY", "langchain") from exc
        chat = ChatOpenAI(model=settings.llm_model)
        return LangChainChatModel(chat, timeout_s=settings.llm_timeout_s)
    return EchoContextModel()


def _embedder(settings: Settings) -> EmbedderPort:
    # Escolha explícita: ter a chave da OpenAI não troca o embedder sozinho.
    if settings.embeddings == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings

            from aiplatform.infrastructure.langchain_embeddings import LangChainEmbedder
        except ImportError as exc:
            raise _missing_extra("EMBEDDINGS=openai", "langchain") from exc
        return LangChainEmbedder(OpenAIEmbeddings(model=settings.embedding_model))
    return HashingEmbedder()


def build_app(settings: Settings) -> FastAPI:
    return create_app(
        _model(settings),
        _embedder(settings),
        _store(settings),
        InMemoryEventBus(),
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
    )


configure_logging()
app = build_app(Settings.from_env())
