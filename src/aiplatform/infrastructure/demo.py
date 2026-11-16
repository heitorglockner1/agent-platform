"""Modelo de chat para a demo sem chave de API.

Não é um modelo: é um roteiro fixo que exercita os dois fluxos da API.
"""

import json
import re
from collections.abc import Sequence

from aiplatform.application.ports import ChatModelPort
from aiplatform.application.rag import CONTEXT_HEADER, NO_CONTEXT_REPLY, QUESTION_HEADER
from aiplatform.domain.models import AgentMessage, ToolCall, ToolSpec


class DemoChatModel(ChatModelPort):
    """Roteiro:

    * com ferramentas e sem resultado de ferramenta no histórico: chama a primeira
      ferramenta com `{"query": <última mensagem do usuário>}`;
    * com resultado de ferramenta: responde com o `text` do primeiro item, ou
      `NO_CONTEXT_REPLY` se a lista vier vazia ou ilegível;
    * sem ferramentas (fluxo do /ask): devolve o primeiro trecho do contexto.
    """

    async def complete(
        self, messages: Sequence[AgentMessage], tools: Sequence[ToolSpec]
    ) -> AgentMessage:
        tool_results = [m for m in messages if m.role == "tool"]
        if tool_results:
            return AgentMessage("assistant", _first_text(tool_results[-1].content))
        if tools:
            query = next(m.content for m in reversed(messages) if m.role == "user")
            call = ToolCall("demo-1", tools[0].name, {"query": query})
            return AgentMessage("assistant", tool_calls=[call])
        return AgentMessage("assistant", _first_context_chunk(messages[-1].content))


def _first_text(tool_output: str) -> str:
    try:
        items = json.loads(tool_output)
    except json.JSONDecodeError:  # resultado cortado pelo teto de tamanho
        return NO_CONTEXT_REPLY
    if isinstance(items, list) and items and isinstance(items[0], dict):
        return str(items[0].get("text", NO_CONTEXT_REPLY))
    return NO_CONTEXT_REPLY


def _first_context_chunk(prompt: str) -> str:
    context = prompt.split("\n\n" + QUESTION_HEADER, 1)[0].removeprefix(CONTEXT_HEADER + "\n")
    # chunk_text normaliza espaços: nenhum trecho contém "\n\n".
    first = context.split("\n\n", 1)[0]
    return re.sub(r"^\[[^\]]*\] ", "", first)
