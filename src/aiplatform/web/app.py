"""API FastAPI: composição na `create_app`, rotas finas, observabilidade."""

import json
import logging
import time
import uuid
from contextvars import ContextVar

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from aiplatform.application.ports import (
    ChatModelPort,
    EmbedderPort,
    EventPublisherPort,
    VectorStorePort,
)
from aiplatform.application.rag import AnswerQuestion, IndexDocument
from aiplatform.domain.models import Document

request_id: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    """Log estruturado: uma linha JSON, sempre com o request_id."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {"level": record.levelname, "msg": record.getMessage(), "request_id": request_id.get()}
        )


class IndexRequest(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)
    source: str = ""


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def create_app(
    model: ChatModelPort,
    embedder: EmbedderPort,
    store: VectorStorePort,
    events: EventPublisherPort,
    registry: CollectorRegistry | None = None,
) -> FastAPI:
    registry = registry or CollectorRegistry()
    requests_total = Counter(
        "http_requests_total", "Requisições", ["path", "status"], registry=registry
    )
    latency = Histogram("http_request_seconds", "Latência", ["path"], registry=registry)
    log = logging.getLogger("aiplatform")

    index_document = IndexDocument(embedder, store, events)
    answer_question = AnswerQuestion(model, embedder, store)
    app = FastAPI(title="AI Agent Platform")

    @app.middleware("http")
    async def observe(req: Request, call_next):
        token = request_id.set(req.headers.get("x-request-id") or uuid.uuid4().hex[:12])
        started = time.perf_counter()
        resp = await call_next(req)
        path = req.scope["route"].path if "route" in req.scope else "unmatched"
        latency.labels(path).observe(time.perf_counter() - started)
        requests_total.labels(path, str(resp.status_code)).inc()
        resp.headers["x-request-id"] = request_id.get()
        log.info("%s %s -> %s", req.method, req.url.path, resp.status_code)
        request_id.reset(token)
        return resp

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/documents", status_code=201)
    def post_document(body: IndexRequest) -> dict:
        n = index_document(Document(body.id, body.text, body.source))
        return {"document_id": body.id, "chunks": n}

    @app.post("/ask")
    async def ask(body: AskRequest) -> dict:
        a = await answer_question(body.question)
        return {
            "answer": a.text,
            "grounded": a.grounded,
            "citations": [c.__dict__ for c in a.citations],
        }

    return app
