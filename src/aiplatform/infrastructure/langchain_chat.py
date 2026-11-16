"""Adaptador LangChain para `ChatModelPort` (extra: `pip install .[langchain]`).

Qualquer `BaseChatModel` do LangChain serve (ChatOpenAI, ChatBedrockConverse,
ChatOllama...), então trocar de provedor não toca em domínio nem aplicação.
O loop do agente continua nosso: o LangChain entra só como camada de provedor.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, assert_never
from uuid import uuid4

from aiplatform.application.ports import ChatModelPort
from aiplatform.domain.models import AgentMessage, TokenUsage, ToolCall, ToolSpec

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage


def _to_lc(messages: Sequence[AgentMessage]) -> list["BaseMessage"]:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    out: list[BaseMessage] = []
    for m in messages:
        match m.role:
            case "system":
                out.append(SystemMessage(content=m.content))
            case "user":
                out.append(HumanMessage(content=m.content))
            case "tool":
                out.append(ToolMessage(content=m.content, tool_call_id=m.tool_call_id or ""))
            case "assistant":
                calls = [{"id": c.id, "name": c.name, "args": c.arguments} for c in m.tool_calls]
                out.append(AIMessage(content=m.content, tool_calls=calls))
            case _:
                assert_never(m.role)
    return out


def _text(content: str | list[str | dict[str, Any]]) -> str:
    """Bedrock e Anthropic devolvem lista de blocos; só os de texto interessam."""
    if isinstance(content, str):
        return content
    parts = [
        block if isinstance(block, str) else str(block.get("text", ""))
        for block in content
        if isinstance(block, str) or block.get("type") == "text"
    ]
    return "".join(parts)


def _tool_schema(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


class LangChainChatModel(ChatModelPort):
    def __init__(self, chat_model: "BaseChatModel") -> None:
        self._chat = chat_model

    async def complete(
        self, messages: Sequence[AgentMessage], tools: Sequence[ToolSpec]
    ) -> AgentMessage:
        # Lista vazia: não chama bind_tools (várias APIs recusam `tools=[]`).
        runnable = self._chat.bind_tools([_tool_schema(t) for t in tools]) if tools else self._chat
        reply = await runnable.ainvoke(_to_lc(messages))
        # Sem id a mensagem `tool` seguinte não teria com o que se correlacionar.
        calls = [
            ToolCall(c["id"] or uuid4().hex, c["name"], c["args"])
            for c in (getattr(reply, "tool_calls", None) or [])
        ]
        meta = getattr(reply, "usage_metadata", None)
        usage = TokenUsage(meta["input_tokens"], meta["output_tokens"]) if meta else None
        return AgentMessage(
            role="assistant", content=_text(reply.content), tool_calls=calls, usage=usage
        )
