"""Entrypoint: `uvicorn aiplatform.main:app`. Sem chave, roda com adaptadores locais."""

import logging
import os

from aiplatform.application.ports import ChatModelPort, VectorStorePort
from aiplatform.infrastructure.fakes import EchoContextModel
from aiplatform.infrastructure.local import HashingEmbedder, InMemoryEventBus, InMemoryVectorStore
from aiplatform.web.app import JsonFormatter, create_app


def _store() -> VectorStorePort:
    if os.environ.get("VECTOR_STORE") == "chroma":
        from aiplatform.infrastructure.chroma_store import ChromaVectorStore

        return ChromaVectorStore(path=os.environ.get("CHROMA_PATH"))
    return InMemoryVectorStore()


def _model() -> ChatModelPort:
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        from aiplatform.infrastructure.langchain_chat import LangChainChatModel

        return LangChainChatModel(ChatOpenAI(model=os.environ.get("LLM_MODEL", "gpt-4o-mini")))
    return EchoContextModel()


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.getLogger("aiplatform").addHandler(handler)
logging.getLogger("aiplatform").setLevel(logging.INFO)

app = create_app(_model(), HashingEmbedder(), _store(), InMemoryEventBus())
