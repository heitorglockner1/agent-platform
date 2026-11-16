"""Rotas finas: validam a entrada, chamam um caso de uso e convertem a saída."""

from dataclasses import dataclass

from fastapi import APIRouter, Response

from aiplatform.application.agentic_rag import AnswerWithAgent
from aiplatform.application.rag import AnswerQuestion, IndexDocument
from aiplatform.web.metrics import ApiMetrics
from aiplatform.web.schemas import (
    AgentRequest,
    AgentResponse,
    AskRequest,
    AskResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
)


@dataclass(frozen=True)
class UseCases:
    """Os casos de uso que a API expõe, já montados com seus adaptadores."""

    index_document: IndexDocument
    answer_question: AnswerQuestion
    answer_with_agent: AnswerWithAgent


def build_router(use_cases: UseCases, metrics: ApiMetrics) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @router.get("/metrics")
    def prometheus() -> Response:
        return Response(metrics.render(), media_type=ApiMetrics.CONTENT_TYPE)

    @router.post("/documents", status_code=201)
    async def index_document(body: IndexRequest) -> IndexResponse:
        """Indexa (ou reindexa) um documento e diz em quantos trechos ele virou."""
        chunks = await use_cases.index_document(body.to_document())
        return IndexResponse(document_id=body.id, chunks=chunks)

    @router.post("/ask")
    async def ask(body: AskRequest) -> AskResponse:
        """Pipeline de RAG: recupera, e só chama o modelo se algo passou do limiar."""
        return AskResponse.from_answer(await use_cases.answer_question(body.question))

    @router.post("/agent")
    async def agent(body: AgentRequest) -> AgentResponse:
        """Agente que decide o que buscar, dentro dos tetos de voltas, tokens e retorno."""
        result = await use_cases.answer_with_agent(body.task)
        metrics.count_agent_run(result.stopped_by)
        return AgentResponse.from_result(result)

    return router
