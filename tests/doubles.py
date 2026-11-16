"""Dublês de teste: modelo de chat com respostas roteirizadas."""

from collections.abc import Sequence
from dataclasses import dataclass

from aiplatform.application.ports import ChatModelPort
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
