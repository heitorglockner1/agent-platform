from aiplatform.application.agent_loop import run_agent
from aiplatform.domain.models import AgentMessage, ToolCall, ToolSpec
from aiplatform.infrastructure.fakes import ScriptedChatModel, TokenBudget


def _call(name="lookup", args=None, id="1"):
    return AgentMessage("assistant", tool_calls=[ToolCall(id, name, args or {"q": "a"})])


def _tool(handler):
    return ToolSpec("lookup", "busca", {"type": "object"}, handler)


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
    async def lookup(q):
        return "ok"

    model = ScriptedChatModel([_call(args={"q": str(i)}, id=str(i)) for i in range(3)])
    r = await run_agent(model, "sys", "go", [_tool(lookup)], max_iterations=3)
    assert model.calls[-1]["tools"] == []
    assert r.iterations == 3


async def test_budget_stops_before_calling_model():
    model = ScriptedChatModel([AgentMessage("assistant", "x")])
    r = await run_agent(model, "sys", "go", [], budget=TokenBudget(0))
    assert r.stopped_by == "budget" and model.calls == []


async def test_tool_result_is_capped():
    async def lookup(q):
        return "x" * 10_000

    model = ScriptedChatModel([_call(), AgentMessage("assistant", "fim")])
    await run_agent(model, "sys", "go", [_tool(lookup)], tool_result_cap=100)
    # o modelo recebe system+user+assistant+tool(capado) na segunda volta
    assert model.calls[1]["n_messages"] == 4
