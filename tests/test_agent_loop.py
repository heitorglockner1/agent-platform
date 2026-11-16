from aiplatform.application.agent_loop import NO_CONCLUSION_REPLY, run_agent
from aiplatform.domain.models import AgentMessage, TokenUsage, ToolCall, ToolSpec
from aiplatform.infrastructure.fakes import ScriptedChatModel, TokenBudget


def _call(name="lookup", args=None, id="1"):
    return AgentMessage("assistant", tool_calls=[ToolCall(id, name, args or {"q": "a"})])


def _tool(handler):
    return ToolSpec("lookup", "busca", {"type": "object"}, handler)


def _tool_messages(model, call_index):
    return [m for m in model.calls[call_index].messages if m.role == "tool"]


async def test_concludes_and_footer_lists_what_really_ran():
    async def lookup(q):
        return {"hit": q}

    model = ScriptedChatModel([_call(), AgentMessage("assistant", "pronto")])
    r = await run_agent(model, "sys", "go", [_tool(lookup)])
    assert r.stopped_by == "concluded"
    assert "[executado de fato: lookup]" in r.final_text


async def test_model_claim_is_not_trusted_when_tool_failed():
    async def lookup(q):
        raise ValueError("fora da janela")

    model = ScriptedChatModel([_call(), AgentMessage("assistant", "criei 3 rascunhos")])
    r = await run_agent(model, "sys", "go", [_tool(lookup)])
    assert "executado de fato: nenhuma" in r.final_text
    assert r.errors and "fora da janela" in r.errors[0]


async def test_repeated_call_does_not_reexecute():
    runs = []

    async def lookup(q):
        runs.append(q)
        return "ok"

    model = ScriptedChatModel([_call(id="1"), _call(id="2"), AgentMessage("assistant", "fim")])
    await run_agent(model, "sys", "go", [_tool(lookup)])
    assert runs == ["a"]


async def test_last_turn_has_no_tools_so_model_must_conclude():
    runs = []

    async def lookup(q):
        runs.append(q)
        return "ok"

    model = ScriptedChatModel([_call(args={"q": str(i)}, id=str(i)) for i in range(3)])
    r = await run_agent(model, "sys", "go", [_tool(lookup)], max_iterations=3)
    assert model.calls[-1].tools == []
    assert runs == ["0", "1"]  # o pedido da última volta não executa
    assert r.stopped_by == "max_iterations" and r.iterations == 3
    assert r.final_text.startswith(NO_CONCLUSION_REPLY)


async def test_budget_stops_before_calling_model():
    model = ScriptedChatModel([AgentMessage("assistant", "x")])
    r = await run_agent(model, "sys", "go", [], budget=TokenBudget(0))
    assert r.stopped_by == "budget" and model.calls == []


async def test_tool_result_is_capped():
    async def lookup(q):
        return "x" * 10_000

    model = ScriptedChatModel([_call(), AgentMessage("assistant", "fim")])
    await run_agent(model, "sys", "go", [_tool(lookup)], tool_result_cap=100)
    [tool_msg] = _tool_messages(model, 1)
    assert "cortado" in tool_msg.content and len(tool_msg.content) < 200


async def test_large_error_is_capped_too():
    async def lookup(q):
        raise ValueError("x" * 10_000)

    model = ScriptedChatModel([_call(), AgentMessage("assistant", "fim")])
    await run_agent(model, "sys", "go", [_tool(lookup)], tool_result_cap=100)
    [tool_msg] = _tool_messages(model, 1)
    assert tool_msg.content.startswith("ERRO:") and len(tool_msg.content) < 200


async def test_tool_result_keeps_accents():
    async def lookup(q):
        return {"msg": "não é possível"}

    model = ScriptedChatModel([_call(), AgentMessage("assistant", "fim")])
    await run_agent(model, "sys", "go", [_tool(lookup)])
    [tool_msg] = _tool_messages(model, 1)
    assert "não é possível" in tool_msg.content


def _calls_with_usage(n, usage=None):
    replies = [_call(args={"q": str(i)}, id=str(i)) for i in range(n)]
    for reply in replies:
        reply.usage = usage
    return replies


async def _ok(q):
    return "ok"


async def test_reported_usage_consumes_the_budget():
    model = ScriptedChatModel(_calls_with_usage(6, TokenUsage(60, 40)))
    r = await run_agent(model, "sys", "go", [_tool(_ok)], max_iterations=6, budget=TokenBudget(150))
    assert r.stopped_by == "budget" and r.iterations == 2
    assert r.tokens_used == 200 and len(model.calls) == 2


async def test_budget_uses_an_estimate_when_provider_reports_nothing():
    model = ScriptedChatModel(_calls_with_usage(6))
    r = await run_agent(model, "s" * 400, "go", [_tool(_ok)], budget=TokenBudget(150))
    assert r.stopped_by == "budget" and r.iterations == 2
