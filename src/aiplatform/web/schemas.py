"""Contrato HTTP: corpo das requisições (com validação) e das respostas.

Os modelos de resposta convertem os objetos de domínio com `from_*`, para as
rotas não montarem dicionários à mão e o OpenAPI descrever cada resposta.
"""

from dataclasses import asdict

from pydantic import BaseModel, Field

from aiplatform.domain.models import AgentResult, Answer, Document


class IndexRequest(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)
    source: str = ""

    def to_document(self) -> Document:
        return Document(self.id, self.text, self.source)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class AgentRequest(BaseModel):
    task: str = Field(min_length=1, max_length=2000)


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

    @classmethod
    def from_answer(cls, answer: Answer) -> "AskResponse":
        citations = [CitationOut(**asdict(c)) for c in answer.citations]
        return cls(answer=answer.text, grounded=answer.grounded, citations=citations)


class AgentResponse(BaseModel):
    answer: str
    stopped_by: str
    iterations: int
    tool_calls_executed: int
    tokens_used: int
    errors: list[str]

    @classmethod
    def from_result(cls, result: AgentResult) -> "AgentResponse":
        return cls(
            answer=result.final_text,
            stopped_by=result.stopped_by,
            iterations=result.iterations,
            tool_calls_executed=result.tool_calls_executed,
            tokens_used=result.tokens_used,
            errors=result.errors,
        )
