"""Configuração lida do ambiente num único ponto."""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeVar, cast, get_args

LlmKind = Literal["demo", "openai"]
VectorStoreKind = Literal["memory", "chroma"]

_N = TypeVar("_N", int, float)


def _choice(environ: Mapping[str, str], name: str, allowed: tuple[str, ...], default: str) -> str:
    value = environ.get(name, default)
    if value not in allowed:
        raise ValueError(f"{name}={value!r} inválido; aceitos: {', '.join(allowed)}")
    return value


def _number(environ: Mapping[str, str], name: str, parse: Callable[[str], _N], default: _N) -> _N:
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        return parse(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} inválido; esperado um número") from None


@dataclass(frozen=True)
class Settings:
    llm: LlmKind = "demo"
    llm_model: str = "gpt-4o-mini"
    vector_store: VectorStoreKind = "memory"
    chroma_path: str | None = None
    rag_top_k: int = 4
    rag_min_score: float = 0.2

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> "Settings":
        store = _choice(environ, "VECTOR_STORE", get_args(VectorStoreKind), cls.vector_store)
        return cls(
            llm="openai" if environ.get("OPENAI_API_KEY") else "demo",
            llm_model=environ.get("LLM_MODEL", cls.llm_model),
            vector_store=cast(VectorStoreKind, store),
            chroma_path=environ.get("CHROMA_PATH") or None,
            rag_top_k=_number(environ, "RAG_TOP_K", int, cls.rag_top_k),
            rag_min_score=_number(environ, "RAG_MIN_SCORE", float, cls.rag_min_score),
        )
