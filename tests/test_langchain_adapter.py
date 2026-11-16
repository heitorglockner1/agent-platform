import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage  # noqa: E402

from aiplatform.domain.models import AgentMessage, ToolSpec  # noqa: E402
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
