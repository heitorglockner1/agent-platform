"""Modelo de chat roteirizado: respostas pré-definidas, para testes e demo."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from aiplatform.application.ports import BudgetPort, ChatModelPort
from aiplatform.application.rag import CONTEXT_HEADER, QUESTION_HEADER
from aiplatform.domain.models import AgentMessage, ToolSpec


@dataclass(frozen=True)
class ModelCall:
    """O que o modelo recebeu numa volta, para asserções."""

    messages: list[AgentMessage]
    tools: list[str]


class ScriptedChatModel(ChatModelPort):
    def __init__(self, replies: list[AgentMessage]) -> None:
        self._replies = list(replies)
        self.calls: list[ModelCall] = []

    async def complete(
        self, messages: Sequence[AgentMessage], tools: Sequence[ToolSpec]
    ) -> AgentMessage:
        # Cópia: o loop continua mutando o histórico depois desta chamada.
        self.calls.append(ModelCall(list(messages), [t.name for t in tools]))
        return self._replies.pop(0) if self._replies else AgentMessage("assistant", "fim")


class EchoContextModel(ChatModelPort):
    """Demo sem chave: devolve o primeiro trecho do contexto recuperado."""

    async def complete(
        self, messages: Sequence[AgentMessage], tools: Sequence[ToolSpec]
    ) -> AgentMessage:
        body = messages[-1].content
        context = body.split("\n\n" + QUESTION_HEADER, 1)[0].removeprefix(CONTEXT_HEADER + "\n")
        # chunk_text normaliza espaços: nenhum trecho contém "\n\n".
        first = context.split("\n\n", 1)[0]
        return AgentMessage("assistant", re.sub(r"^\[[^\]]*\] ", "", first))


class TokenBudget(BudgetPort):
    def __init__(self, max_tokens: int) -> None:
        self._left = max_tokens

    def allows_next_call(self) -> bool:
        return self._left > 0

    def record(self, tokens: int) -> None:
        self._left -= tokens
