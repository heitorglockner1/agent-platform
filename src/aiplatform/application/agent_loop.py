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
from collections.abc import Sequence
from dataclasses import dataclass

from aiplatform.application.ports import BudgetPort, ChatModelPort, Tracer
from aiplatform.domain.models import AgentMessage, AgentResult, StopReason, ToolCall, ToolSpec

NO_CONCLUSION_REPLY = "Não foi possível concluir dentro dos limites definidos."


@dataclass(frozen=True)
class _ToolOutcome:
    content: str
    ok: bool


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [cortado: {len(text) - limit} caracteres]"


def _footer(executed: list[str], errors: list[str]) -> str:
    done = ", ".join(executed) if executed else "nenhuma"
    line = f"\n\n[executado de fato: {done}"
    if errors:
        line += f"; recusado/erro: {len(errors)}"
    return line + "]"


def _estimate_tokens(sent: Sequence[AgentMessage], reply: AgentMessage) -> int:
    """Aproximação (4 caracteres por token), usada só quando o provedor não informa o uso."""
    return max(1, (sum(len(m.content) for m in sent) + len(reply.content)) // 4)


async def _run_tool(call: ToolCall, specs: dict[str, ToolSpec], cap: int) -> _ToolOutcome:
    spec = specs.get(call.name)
    if spec is None:
        return _ToolOutcome(f"ERRO: ferramenta desconhecida '{call.name}'.", ok=False)
    try:
        result = await spec.handler(**call.arguments)
    except Exception as exc:  # o handler é lei: recusa vira texto para o modelo
        # Exceção pode carregar um corpo HTTP inteiro: o teto vale também aqui.
        return _ToolOutcome(_truncate(f"ERRO: {type(exc).__name__}: {exc}", cap), ok=False)
    return _ToolOutcome(_truncate(json.dumps(result, default=str, ensure_ascii=False), cap), True)


async def run_agent(
    model: ChatModelPort,
    system_prompt: str,
    user_prompt: str,
    tools: Sequence[ToolSpec],
    *,
    max_iterations: int = 6,
    tool_result_cap: int = 4000,
    budget: BudgetPort | None = None,
    tracer: Tracer | None = None,
) -> AgentResult:
    specs = {t.name: t for t in tools}
    history = [
        AgentMessage(role="system", content=system_prompt),
        AgentMessage(role="user", content=user_prompt),
    ]
    seen: set[str] = set()
    executed: list[str] = []
    errors: list[str] = []
    tokens_used = 0

    for turn in range(1, max_iterations + 1):
        if budget is not None and not budget.allows_next_call():
            return _finish(history, turn - 1, executed, errors, "budget", tokens_used)

        last_turn = turn == max_iterations
        started = time.perf_counter()
        sent = list(history)
        reply = await model.complete(sent, [] if last_turn else list(tools))
        tokens = reply.usage.total if reply.usage else _estimate_tokens(sent, reply)
        tokens_used += tokens
        if budget is not None:
            budget.record(tokens)
        if tracer:
            tracer.event(
                "agent.turn",
                turn=turn,
                tool_calls=len(reply.tool_calls),
                tokens=tokens,
                ms=round((time.perf_counter() - started) * 1000),
            )
        history.append(reply)
        if not reply.tool_calls:
            return _finish(history, turn, executed, errors, "concluded", tokens_used)
        if last_turn:
            # Nenhuma ferramenta foi oferecida: o pedido não é atendido.
            break

        for call in reply.tool_calls:
            key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
            if key in seen:
                out = "Chamada repetida: o resultado já está acima. Conclua com o que tem."
            else:
                seen.add(key)
                outcome = await _run_tool(call, specs, tool_result_cap)
                out = outcome.content
                if outcome.ok:
                    executed.append(call.name)
                else:
                    errors.append(out)
            history.append(
                AgentMessage(role="tool", content=out, tool_call_id=call.id, name=call.name)
            )

    return _finish(history, max_iterations, executed, errors, "max_iterations", tokens_used)


def _finish(
    history: list[AgentMessage],
    turns: int,
    executed: list[str],
    errors: list[str],
    reason: StopReason,
    tokens_used: int,
) -> AgentResult:
    last = next(
        (m.content for m in reversed(history) if m.role == "assistant" and m.content),
        NO_CONCLUSION_REPLY,
    )
    return AgentResult(
        last + _footer(executed, errors), turns, len(executed), reason, errors, tokens_used
    )
