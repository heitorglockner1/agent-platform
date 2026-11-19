"""RAG com agente: o modelo decide o que buscar, o loop impõe os tetos."""

from dataclasses import dataclass
from functools import partial
from typing import Any

from aiplatform.application.agent_loop import run_agent
from aiplatform.application.budget import TokenBudget
from aiplatform.application.ports import ChatModelPort, Tracer
from aiplatform.application.prompts import (
    AGENT_SYSTEM_PROMPT,
    SEARCH_TOOL_DESCRIPTION,
    SEARCH_TOOL_NAME,
    SEARCH_TOOL_SCHEMA,
)
from aiplatform.application.rag import Retriever
from aiplatform.domain.models import AgentResult, TenantId, ToolSpec


@dataclass(frozen=True)
class AgentLimits:
    """Tetos de uma execução: voltas, tokens e tamanho do retorno de ferramenta."""

    max_iterations: int = 4
    max_tokens: int = 20_000
    tool_result_cap: int = 6_000


DEFAULT_LIMITS = AgentLimits()


class AnswerWithAgent:
    """Caso de uso: agente com uma única ferramenta, a busca nos documentos."""

    def __init__(
        self,
        model: ChatModelPort,
        retrieve: Retriever,
        limits: AgentLimits = DEFAULT_LIMITS,
        tracer: Tracer | None = None,
    ) -> None:
        self._model = model
        self._retrieve = retrieve
        self._limits = limits
        self._tracer = tracer

    async def __call__(self, tenant: TenantId, task: str) -> AgentResult:
        return await run_agent(
            self._model,
            AGENT_SYSTEM_PROMPT,
            task,
            [self._search_tool(tenant)],
            max_iterations=self._limits.max_iterations,
            tool_result_cap=self._limits.tool_result_cap,
            budget=TokenBudget(self._limits.max_tokens),  # novo a cada execução
            tracer=self._tracer,
        )

    def _search_tool(self, tenant: TenantId) -> ToolSpec:
        """Ferramenta presa ao cliente da execução: o modelo não escolhe de quem busca."""
        handler = partial(self._search_documents, tenant)
        return ToolSpec(SEARCH_TOOL_NAME, SEARCH_TOOL_DESCRIPTION, SEARCH_TOOL_SCHEMA, handler)

    async def _search_documents(self, tenant: TenantId, query: str) -> list[dict[str, Any]]:
        """Handler da ferramenta: lista vazia quando nada passa do limiar."""
        return [
            {
                "chunk_id": h.chunk.id,
                "source": h.chunk.source,
                "text": h.chunk.text,
                "score": round(h.score, 4),
            }
            for h in await self._retrieve(tenant, query)
        ]
