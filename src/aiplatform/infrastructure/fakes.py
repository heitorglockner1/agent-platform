"""Modelo de chat roteirizado: respostas pré-definidas, para testes e demo."""

from typing import List, Sequence

from aiplatform.application.ports import BudgetPort, ChatModelPort
from aiplatform.domain.models import AgentMessage, ToolSpec


class ScriptedChatModel(ChatModelPort):
    def __init__(self, replies: List[AgentMessage]) -> None:
        self._replies = list(replies)
        self.calls: List[dict] = []  # o que o modelo recebeu, para asserções

    async def complete(
        self, messages: Sequence[AgentMessage], tools: Sequence[ToolSpec]
    ) -> AgentMessage:
        self.calls.append({"n_messages": len(messages), "tools": [t.name for t in tools]})
        return self._replies.pop(0) if self._replies else AgentMessage("assistant", "fim")


class EchoContextModel(ChatModelPort):
    """Demo sem chave: devolve o primeiro trecho do contexto recuperado."""

    async def complete(
        self, messages: Sequence[AgentMessage], tools: Sequence[ToolSpec]
    ) -> AgentMessage:
        body = messages[-1].content
        first = body.split("\n\n")[1] if "\n\n" in body else body
        return AgentMessage("assistant", first)


class TokenBudget(BudgetPort):
    def __init__(self, max_tokens: int) -> None:
        self._left = max_tokens

    def allows_next_call(self) -> bool:
        return self._left > 0

    def record(self, tokens: int) -> None:
        self._left -= tokens
