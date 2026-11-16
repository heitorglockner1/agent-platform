"""Montagem da API: casos de uso, rotas, middleware e tratamento de erros.

Cada parte mora no seu módulo: `schemas` (contrato HTTP), `routes` (rotas
finas), `middleware` (request id, métricas, erros) e `metrics` (Prometheus).
"""

from fastapi import FastAPI
from prometheus_client import CollectorRegistry

from aiplatform.application.agentic_rag import DEFAULT_LIMITS, AgentLimits, AnswerWithAgent
from aiplatform.application.ports import (
    ChatModelPort,
    EmbedderPort,
    EventPublisherPort,
    ProviderUnavailableError,
    Tracer,
    VectorStorePort,
)
from aiplatform.application.rag import AnswerQuestion, IndexDocument, Retriever
from aiplatform.web.metrics import ApiMetrics
from aiplatform.web.middleware import RequestObserver, provider_unavailable
from aiplatform.web.routes import UseCases, build_router


def create_app(
    model: ChatModelPort,
    embedder: EmbedderPort,
    store: VectorStorePort,
    events: EventPublisherPort,
    registry: CollectorRegistry | None = None,
    *,
    top_k: int = 4,
    min_score: float = 0.2,
    agent_limits: AgentLimits = DEFAULT_LIMITS,
    tracer: Tracer | None = None,
) -> FastAPI:
    """Monta a aplicação a partir dos adaptadores; não lê ambiente nem configuração."""
    # /ask e /agent compartilham o mesmo Retriever: mesmo top_k, limiar e trace.
    retrieve = Retriever(embedder, store, top_k=top_k, min_score=min_score, tracer=tracer)
    use_cases = UseCases(
        index_document=IndexDocument(embedder, store, events),
        answer_question=AnswerQuestion(model, retrieve),
        answer_with_agent=AnswerWithAgent(model, retrieve, agent_limits, tracer),
    )
    metrics = ApiMetrics(registry)

    app = FastAPI(title="Agent Platform")
    app.add_exception_handler(ProviderUnavailableError, provider_unavailable)
    app.middleware("http")(RequestObserver(metrics))
    app.include_router(build_router(use_cases, metrics))
    return app
