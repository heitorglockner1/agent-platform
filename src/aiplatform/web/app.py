"""API FastAPI: composição na `create_app`, rotas finas, observabilidade."""

import logging
import re
import time
import uuid
from dataclasses import asdict

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field
from starlette.middleware.base import RequestResponseEndpoint

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
from aiplatform.domain.models import Document
from aiplatform.observability import request_id

# O id vem do cliente e vai para os logs: só aceita um formato curto e seguro.
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class IndexRequest(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)
    source: str = ""


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class AgentRequest(BaseModel):
    task: str = Field(min_length=1, max_length=2000)


class AgentResponse(BaseModel):
    answer: str
    stopped_by: str
    iterations: int
    tool_calls_executed: int
    tokens_used: int
    errors: list[str]


class HealthResponse(BaseModel):
    status: str


class IndexResponse(BaseModel):
    document_id: str
    chunks: int


class CitationOut(BaseModel):
    chunk_id: str
    source: str
    score: float


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    citations: list[CitationOut]


def _incoming_request_id(req: Request) -> str:
    candidate = req.headers.get("x-request-id", "")
    return candidate if _REQUEST_ID.match(candidate) else uuid.uuid4().hex[:12]


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
    registry = registry or CollectorRegistry()
    requests_total = Counter(
        "http_requests_total", "Requisições", ["path", "status"], registry=registry
    )
    latency = Histogram("http_request_seconds", "Latência", ["path"], registry=registry)
    agent_runs = Counter(
        "agent_runs_total", "Execuções do agente", ["stopped_by"], registry=registry
    )
    log = logging.getLogger("aiplatform")

    index_document = IndexDocument(embedder, store, events)
    retrieve = Retriever(embedder, store, top_k=top_k, min_score=min_score, tracer=tracer)
    answer_question = AnswerQuestion(model, retrieve)
    answer_with_agent = AnswerWithAgent(model, retrieve, agent_limits, tracer)
    app = FastAPI(title="Agent Platform")

    @app.exception_handler(ProviderUnavailableError)
    async def provider_unavailable(_: Request, exc: ProviderUnavailableError) -> JSONResponse:
        log.warning("provedor indisponível: %s", exc)
        return JSONResponse({"detail": "Provedor de IA indisponível."}, status_code=502)

    @app.middleware("http")
    async def observe(req: Request, call_next: RequestResponseEndpoint) -> Response:
        token = request_id.set(_incoming_request_id(req))
        started = time.perf_counter()
        status = 500
        try:
            resp = await call_next(req)
            status = resp.status_code
        except Exception:
            log.exception("erro não tratado em %s %s", req.method, req.url.path)
            resp = JSONResponse(
                {"detail": "Erro interno.", "request_id": request_id.get()}, status_code=500
            )
        finally:
            path = req.scope["route"].path if "route" in req.scope else "unmatched"
            latency.labels(path).observe(time.perf_counter() - started)
            requests_total.labels(path, str(status)).inc()
            log.info("%s %s -> %s", req.method, req.url.path, status)
            resp.headers["x-request-id"] = request_id.get()
            request_id.reset(token)
        return resp

    @app.get("/health")
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/documents", status_code=201)
    async def post_document(body: IndexRequest) -> IndexResponse:
        n = await index_document(Document(body.id, body.text, body.source))
        return IndexResponse(document_id=body.id, chunks=n)

    @app.post("/ask")
    async def ask(body: AskRequest) -> AskResponse:
        a = await answer_question(body.question)
        return AskResponse(
            answer=a.text,
            grounded=a.grounded,
            citations=[CitationOut(**asdict(c)) for c in a.citations],
        )

    @app.post("/agent")
    async def agent(body: AgentRequest) -> AgentResponse:
        r = await answer_with_agent(body.task)
        agent_runs.labels(r.stopped_by).inc()
        return AgentResponse(
            answer=r.final_text,
            stopped_by=r.stopped_by,
            iterations=r.iterations,
            tool_calls_executed=r.tool_calls_executed,
            tokens_used=r.tokens_used,
            errors=r.errors,
        )

    return app
