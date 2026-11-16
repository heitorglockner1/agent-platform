"""Modelo de chat para a demo sem chave de API."""

import re
from collections.abc import Sequence

from aiplatform.application.ports import ChatModelPort
from aiplatform.application.rag import CONTEXT_HEADER, QUESTION_HEADER
from aiplatform.domain.models import AgentMessage, ToolSpec


class EchoContextModel(ChatModelPort):
    """Devolve o primeiro trecho do contexto recuperado."""

    async def complete(
        self, messages: Sequence[AgentMessage], tools: Sequence[ToolSpec]
    ) -> AgentMessage:
        body = messages[-1].content
        context = body.split("\n\n" + QUESTION_HEADER, 1)[0].removeprefix(CONTEXT_HEADER + "\n")
        # chunk_text normaliza espaços: nenhum trecho contém "\n\n".
        first = context.split("\n\n", 1)[0]
        return AgentMessage("assistant", re.sub(r"^\[[^\]]*\] ", "", first))
