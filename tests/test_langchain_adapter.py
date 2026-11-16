import asyncio

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402

from aiplatform.application.ports import ProviderUnavailableError  # noqa: E402
from aiplatform.domain.models import AgentMessage, ToolCall, ToolSpec  # noqa: E402
from aiplatform.infrastructure.langchain_chat import LangChainChatModel  # noqa: E402


class StubChat:
    """Imita BaseChatModel: bind_tools devolve um runnable com ainvoke."""

    def __init__(self):
        self.bound = None
        self.seen = None

    def bind_tools(self, tools):
        self.bound = tools
        return self

    async def ainvoke(self, messages):
        self.seen = messages
        return AIMessage(
            content="", tool_calls=[{"id": "c1", "name": "lookup", "args": {"q": "x"}}]
        )


async def _noop(**_):
    return None


async def test_tools_become_tool_calls():
    stub = StubChat()
    spec = ToolSpec("lookup", "busca", {"type": "object"}, _noop)
    reply = await LangChainChatModel(stub).complete([AgentMessage("user", "oi")], [spec])
    assert stub.bound[0]["function"]["name"] == "lookup"
    assert reply.tool_calls[0].arguments == {"q": "x"}


async def test_empty_tools_does_not_bind():
    stub = StubChat()
    await LangChainChatModel(stub).complete([AgentMessage("user", "oi")], [])
    assert stub.bound is None


class UsageChat(StubChat):
    async def ainvoke(self, messages):
        usage = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        return AIMessage(content="ok", usage_metadata=usage)


async def test_usage_reported_by_provider_is_kept():
    reply = await LangChainChatModel(UsageChat()).complete([AgentMessage("user", "oi")], [])
    assert reply.usage is not None and reply.usage.total == 15


class BlocksChat(StubChat):
    async def ainvoke(self, messages):
        blocks = [
            {"type": "text", "text": "Olá"},
            {"type": "tool_use", "id": "c1", "name": "lookup", "input": {}},
        ]
        return AIMessage(content=blocks, tool_calls=[{"id": None, "name": "lookup", "args": {}}])


async def test_content_blocks_become_plain_text_and_missing_id_is_filled():
    reply = await LangChainChatModel(BlocksChat()).complete([AgentMessage("user", "oi")], [])
    assert reply.content == "Olá"
    assert reply.tool_calls[0].id


async def test_every_role_maps_to_its_langchain_message():
    stub = StubChat()
    history = [
        AgentMessage("system", "s"),
        AgentMessage("user", "u"),
        AgentMessage("assistant", tool_calls=[ToolCall("c1", "lookup", {})]),
        AgentMessage("tool", "r", tool_call_id="c1"),
    ]
    await LangChainChatModel(stub).complete(history, [])
    assert [type(m) for m in stub.seen] == [SystemMessage, HumanMessage, AIMessage, ToolMessage]


class SlowChat(StubChat):
    async def ainvoke(self, messages):
        await asyncio.sleep(1)


async def test_slow_provider_times_out_as_unavailable():
    with pytest.raises(ProviderUnavailableError):
        await LangChainChatModel(SlowChat(), timeout_s=0.01).complete(
            [AgentMessage("user", "oi")], []
        )
