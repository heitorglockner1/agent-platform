"""Adaptador LangChain para `ChatModelPort` (extra: `pip install .[langchain]`).

Qualquer `BaseChatModel` do LangChain serve (ChatOpenAI, ChatBedrockConverse,
ChatOllama...), então trocar de provedor não toca em domínio nem aplicação.
O loop do agente continua nosso: o LangChain entra só como camada de provedor.
"""

import json
from typing import Any, Dict, List, Sequence

from aiplatform.application.ports import ChatModelPort
from aiplatform.domain.models import AgentMessage, ToolCall, ToolSpec


def _to_lc(messages: Sequence[AgentMessage]) -> List[Any]:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    out: List[Any] = []
    for m in messages:
        if m.role == "system":
            out.append(SystemMessage(content=m.content))
        elif m.role == "user":
            out.append(HumanMessage(content=m.content))
        elif m.role == "tool":
            out.append(ToolMessage(content=m.content, tool_call_id=m.tool_call_id or ""))
        else:
            calls = [{"id": c.id, "name": c.name, "args": c.arguments} for c in m.tool_calls]
            out.append(AIMessage(content=m.content, tool_calls=calls))
    return out


def _tool_schema(spec: ToolSpec) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


class LangChainChatModel(ChatModelPort):
    def __init__(self, chat_model: Any) -> None:
        self._chat = chat_model

    async def complete(
        self, messages: Sequence[AgentMessage], tools: Sequence[ToolSpec]
    ) -> AgentMessage:
        # Lista vazia: não chama bind_tools (várias APIs recusam `tools=[]`).
        runnable = self._chat.bind_tools([_tool_schema(t) for t in tools]) if tools else self._chat
        reply = await runnable.ainvoke(_to_lc(messages))
        calls = [
            ToolCall(
                c["id"],
                c["name"],
                c["args"] if isinstance(c["args"], dict) else json.loads(c["args"]),
            )
            for c in (getattr(reply, "tool_calls", None) or [])
        ]
        return AgentMessage(role="assistant", content=str(reply.content or ""), tool_calls=calls)
