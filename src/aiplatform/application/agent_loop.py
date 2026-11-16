"""Loop de agente com tetos explícitos.

Lições de produção embutidas aqui (ver docs/LEARNINGS.md):

* três tetos sempre: `max_iterations`, `budget` reconferido a cada volta e
  tamanho máximo do retorno de ferramenta (ele é reenviado em TODA volta);
* na última volta o modelo recebe `tools=[]`: sem isso o teto vira um final
  sem saída, porque ele pede mais uma ferramenta em vez de concluir;
* mesma ferramenta + mesmos argumentos não reexecuta: devolve um lembrete;
* o modelo afirma o que não fez, então o texto final leva um rodapé gerado por
  código com o que de fato foi executado;
* erro de ferramenta volta como mensagem ao modelo, não derruba o loop.
"""

import json
import time
from typing import Dict, List, Optional, Sequence, Set

from aiplatform.application.ports import BudgetPort, ChatModelPort, Tracer
from aiplatform.domain.models import AgentMessage, AgentResult, ToolCall, ToolSpec


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [cortado: {len(text) - limit} caracteres]"


def _footer(executed: List[str], errors: List[str]) -> str:
    done = ", ".join(executed) if executed else "nenhuma"
    line = f"\n\n[executado de fato: {done}"
    if errors:
        line += f"; recusado/erro: {len(errors)}"
    return line + "]"


async def _run_tool(call: ToolCall, specs: Dict[str, ToolSpec], cap: int) -> str:
    spec = specs.get(call.name)
    if spec is None:
        return f"ERRO: ferramenta desconhecida '{call.name}'."
    try:
        return _truncate(json.dumps(await spec.handler(**call.arguments), default=str), cap)
    except Exception as exc:  # o handler é lei: recusa vira texto para o modelo
        return f"ERRO: {type(exc).__name__}: {exc}"


async def run_agent(
    model: ChatModelPort,
    system_prompt: str,
    user_prompt: str,
    tools: Sequence[ToolSpec],
    *,
    max_iterations: int = 6,
    tool_result_cap: int = 4000,
    budget: Optional[BudgetPort] = None,
    tracer: Optional[Tracer] = None,
) -> AgentResult:
    specs = {t.name: t for t in tools}
    history = [
        AgentMessage(role="system", content=system_prompt),
        AgentMessage(role="user", content=user_prompt),
    ]
    seen: Set[str] = set()
    executed: List[str] = []
    errors: List[str] = []

    for turn in range(1, max_iterations + 1):
        if budget is not None and not budget.allows_next_call():
            return _finish(history, turn - 1, executed, errors, "budget")

        last_turn = turn == max_iterations
        started = time.perf_counter()
        reply = await model.complete(history, [] if last_turn else list(tools))
        if tracer:
            tracer.event(
                "agent.turn",
                turn=turn,
                tool_calls=len(reply.tool_calls),
                ms=round((time.perf_counter() - started) * 1000),
            )
        history.append(reply)
        if not reply.tool_calls:
            return AgentResult(
                reply.content + _footer(executed, errors), turn, len(executed), "concluded", errors
            )

        for call in reply.tool_calls:
            key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
            if key in seen:
                out = "Chamada repetida: o resultado já está acima. Conclua com o que tem."
            else:
                seen.add(key)
                out = await _run_tool(call, specs, tool_result_cap)
                (errors if out.startswith("ERRO:") else executed).append(
                    out if out.startswith("ERRO:") else call.name
                )
            history.append(
                AgentMessage(role="tool", content=out, tool_call_id=call.id, name=call.name)
            )

    return _finish(history, max_iterations, executed, errors, "max_iterations")


def _finish(
    history: List[AgentMessage], turns: int, executed: List[str], errors: List[str], reason: str
) -> AgentResult:
    last = next((m.content for m in reversed(history) if m.role == "assistant" and m.content), "")
    return AgentResult(last + _footer(executed, errors), turns, len(executed), reason, errors)
