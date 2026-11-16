import json
import logging
import sys

import pytest
from fastapi.testclient import TestClient

from aiplatform.main import build_app
from aiplatform.observability import JsonFormatter
from aiplatform.settings import Settings


def test_settings_defaults():
    assert Settings.from_env({}) == Settings()


def test_settings_read_every_variable():
    s = Settings.from_env(
        {
            "OPENAI_API_KEY": "sk-x",
            "LLM_MODEL": "gpt-x",
            "LLM_TIMEOUT_S": "5",
            "VECTOR_STORE": "chroma",
            "CHROMA_PATH": "/data",
            "EMBEDDINGS": "openai",
            "EMBEDDING_MODEL": "emb-x",
            "RAG_TOP_K": "8",
            "RAG_MIN_SCORE": "0.5",
        }
    )
    assert s == Settings("openai", "gpt-x", 5.0, "chroma", "/data", "openai", "emb-x", 8, 0.5)


@pytest.mark.parametrize(
    "env",
    [
        {"EMBEDDINGS": "bert"},
        {"VECTOR_STORE": "xyz"},
        {"RAG_TOP_K": "muitos"},
        {"RAG_MIN_SCORE": "alto"},
    ],
)
def test_invalid_settings_name_the_variable(env):
    [name] = env
    with pytest.raises(ValueError, match=name):
        Settings.from_env(env)


def test_default_composition_serves_health():
    assert TestClient(build_app(Settings())).get("/health").status_code == 200


def test_json_log_carries_traceback():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        exc_info = sys.exc_info()
    record = logging.LogRecord("aiplatform", logging.ERROR, __file__, 1, "falhou", None, exc_info)
    entry = json.loads(JsonFormatter().format(record))
    assert "RuntimeError: boom" in entry["exc"] and entry["logger"] == "aiplatform"


def test_chroma_composition():
    pytest.importorskip("chromadb")
    assert TestClient(build_app(Settings(vector_store="chroma"))).get("/health").status_code == 200


def test_openai_composition(monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-teste")
    settings = Settings(llm="openai", embeddings="openai")
    assert TestClient(build_app(settings)).get("/health").status_code == 200


def test_missing_extra_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "chromadb", None)  # import passa a falhar
    with pytest.raises(RuntimeError, match=r"pip install \".\[chroma\]\""):
        build_app(Settings(vector_store="chroma"))
