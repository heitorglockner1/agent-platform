"""Configuração lida do ambiente num único ponto."""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeVar, cast, get_args

LlmKind = Literal["demo", "openai"]
VectorStoreKind = Literal["memory", "chroma"]
EmbeddingsKind = Literal["hashing", "openai"]

_N = TypeVar("_N", int, float)


def _choice(environ: Mapping[str, str], name: str, allowed: tuple[str, ...], default: str) -> str:
    """Valor da variável, recusando o que não estiver em `allowed`."""
    value = environ.get(name, default)
    if value not in allowed:
        raise ValueError(f"{name}={value!r} inválido; aceitos: {', '.join(allowed)}")
    return value


def _number(environ: Mapping[str, str], name: str, parse: Callable[[str], _N], default: _N) -> _N:
    """Valor numérico da variável, com erro que diz qual variável está errada."""
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        return parse(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} inválido; esperado um número") from None


@dataclass(frozen=True)
class Settings:
    """Toda a configuração da aplicação. Os padrões rodam sem chave e sem rede."""

    llm: LlmKind = "demo"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_s: float = 30.0
    vector_store: VectorStoreKind = "memory"
    chroma_path: str | None = None
    embeddings: EmbeddingsKind = "hashing"
    embedding_model: str = "text-embedding-3-small"
    rag_top_k: int = 4
    rag_min_score: float = 0.2
    agent_max_iterations: int = 4
    agent_max_tokens: int = 20_000

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> "Settings":
        emb = _choice(environ, "EMBEDDINGS", get_args(EmbeddingsKind), cls.embeddings)
        store = _choice(environ, "VECTOR_STORE", get_args(VectorStoreKind), cls.vector_store)
        return cls(
            llm="openai" if environ.get("OPENAI_API_KEY") else "demo",
            llm_model=environ.get("LLM_MODEL", cls.llm_model),
            llm_timeout_s=_number(environ, "LLM_TIMEOUT_S", float, cls.llm_timeout_s),
            vector_store=cast(VectorStoreKind, store),
            chroma_path=environ.get("CHROMA_PATH") or None,
            embeddings=cast(EmbeddingsKind, emb),
            embedding_model=environ.get("EMBEDDING_MODEL", cls.embedding_model),
            rag_top_k=_number(environ, "RAG_TOP_K", int, cls.rag_top_k),
            rag_min_score=_number(environ, "RAG_MIN_SCORE", float, cls.rag_min_score),
            agent_max_iterations=_number(
                environ, "AGENT_MAX_ITERATIONS", int, cls.agent_max_iterations
            ),
            agent_max_tokens=_number(environ, "AGENT_MAX_TOKENS", int, cls.agent_max_tokens),
        )
