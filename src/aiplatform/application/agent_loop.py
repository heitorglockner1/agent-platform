"""Loop de agente com tetos explícitos.

Lições de produção embutidas aqui (ver docs/LEARNINGS.md):

* três tetos sempre: `max_iterations`, `budget` reconferido a cada volta e
  tamanho máximo do retorno de ferramenta (ele é reenviado em TODA volta);
* o orçamento consome o uso de tokens informado pelo provedor, ou uma
  estimativa quando ele não informa;
* na última volta o modelo recebe `tools=[]`: sem isso o teto vira um final
  sem saída, porque ele pede mais uma ferramenta em vez de concluir. Se pedir
  mesmo assim, o pedido é ignorado;
* mesma ferramenta + mesmos argumentos não reexecuta: devolve um lembrete;
* o modelo afirma o que não fez, então o texto final leva um rodapé gerado por
  código com o que de fato foi executado;
* erro de ferramenta volta como mensagem ao modelo, não derruba o loop.

Organização: `run_agent` só conta as voltas e decide quando parar. Cada
responsabilidade mora numa classe pequena:

* `Conversation`: o histórico enviado ao modelo;
* `TokenMeter`: consumo de tokens e orçamento;
* `ToolRunner`: execução das ferramentas, com dedupe e teto de tamanho;
* `ExecutionLog`: o que de fato rodou, base do rodapé factual;
* `_AgentRun`: junta as quatro numa execução.
"""

import json
import time
from collections.abc import Sequence

from aiplatform.application.ports import BudgetPort, ChatModelPort, Tracer
from aiplatform.application.prompts import REPEATED_CALL_REPLY
from aiplatform.domain.models import AgentMessage, AgentResult, StopReason, ToolCall, ToolSpec

NO_CONCLUSION_REPLY = "Não foi possível concluir dentro dos limites definidos."


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
    """Conversa com o modelo até ele responder em texto ou um teto ser atingido.

    Cada volta: confere o orçamento, chama o modelo e executa as ferramentas
    que ele pediu. A última volta não oferece ferramentas, para forçar uma
    conclusão.
    """
    run = _AgentRun(
        model,
        Conversation(system_prompt, user_prompt),
        ToolRunner(tools, tool_result_cap),
        TokenMeter(budget),
        tracer,
    )
    for turn in range(1, max_iterations + 1):
        if not run.meter.allows_next_call():
            return run.finish(turn - 1, "budget")

        is_last_turn = turn == max_iterations
        reply = await run.ask_model(turn, offer_tools=not is_last_turn)
        if not reply.tool_calls:
            return run.finish(turn, "concluded")
        if is_last_turn:
            break  # pediu ferramenta sem nenhuma oferecida: o pedido não é atendido

        await run.execute(reply.tool_calls)

    return run.finish(max_iterations, "max_iterations")


class Conversation:
    """Histórico de mensagens, na ordem em que o modelo as recebe."""

    def __init__(self, system_prompt: str, user_prompt: str) -> None:
        self._messages = [
            AgentMessage(role="system", content=system_prompt),
            AgentMessage(role="user", content=user_prompt),
        ]

    def snapshot(self) -> list[AgentMessage]:
        """Cópia do histórico: quem recebe pode guardá-la sem ver mudanças futuras."""
        return list(self._messages)

    def add(self, message: AgentMessage) -> None:
        self._messages.append(message)

    def add_tool_result(self, call: ToolCall, content: str) -> None:
        """Responde a um pedido de ferramenta, ligando a resposta ao id do pedido."""
        self.add(AgentMessage(role="tool", content=content, tool_call_id=call.id, name=call.name))

    def last_answer(self) -> str:
        """Último texto do modelo; se ele nunca respondeu em texto, a recusa padrão."""
        answers = (m.content for m in reversed(self._messages) if m.role == "assistant")
        return next((text for text in answers if text), NO_CONCLUSION_REPLY)


class TokenMeter:
    """Soma os tokens de cada volta e os desconta do orçamento, se houver um."""

    def __init__(self, budget: BudgetPort | None) -> None:
        self._budget = budget
        self.used = 0

    def allows_next_call(self) -> bool:
        return self._budget is None or self._budget.allows_next_call()

    def charge(self, sent: Sequence[AgentMessage], reply: AgentMessage) -> int:
        """Registra o custo da volta e o devolve (uso informado ou estimado)."""
        tokens = reply.usage.total if reply.usage else estimate_tokens(sent, reply)
        self.used += tokens
        if self._budget is not None:
            self._budget.record(tokens)
        return tokens


def estimate_tokens(sent: Sequence[AgentMessage], reply: AgentMessage) -> int:
    """Aproximação (4 caracteres por token), usada só quando o provedor não informa o uso."""
    characters = sum(len(m.content) for m in sent) + len(reply.content)
    return max(1, characters // 4)


class ExecutionLog:
    """O que as ferramentas fizeram de fato, independente do que o modelo diz."""

    def __init__(self) -> None:
        self.executed: list[str] = []
        self.errors: list[str] = []

    def footer(self) -> str:
        """Rodapé anexado à resposta final, gerado por código e não pelo modelo."""
        done = ", ".join(self.executed) or "nenhuma"
        refused = f"; recusado/erro: {len(self.errors)}" if self.errors else ""
        return f"\n\n[executado de fato: {done}{refused}]"


class ToolRunner:
    """Executa os pedidos de ferramenta do modelo.

    Erros do handler viram texto para o modelo (o handler é lei: recusar é
    resposta válida), pedidos repetidos não reexecutam e todo retorno é
    cortado em `result_cap` caracteres, porque ele volta ao modelo em cada
    volta seguinte.
    """

    def __init__(self, tools: Sequence[ToolSpec], result_cap: int) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._result_cap = result_cap
        self._seen: set[str] = set()
        self.log = ExecutionLog()

    @property
    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    async def run(self, call: ToolCall) -> str:
        """Executa um pedido e devolve o texto que o modelo vai receber."""
        key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
        if key in self._seen:
            return REPEATED_CALL_REPLY
        self._seen.add(key)

        try:
            output = await self._invoke(call)
        except Exception as exc:
            # A mensagem da exceção pode carregar um corpo HTTP inteiro: corta também.
            error = self._cap(f"ERRO: {type(exc).__name__}: {exc}")
            self.log.errors.append(error)
            return error

        self.log.executed.append(call.name)
        return self._cap(json.dumps(output, default=str, ensure_ascii=False))

    async def _invoke(self, call: ToolCall) -> object:
        tool = self._tools.get(call.name)
        if tool is None:
            raise LookupError(f"ferramenta desconhecida '{call.name}'")
        return await tool.handler(**call.arguments)

    def _cap(self, text: str) -> str:
        if len(text) <= self._result_cap:
            return text
        cut = len(text) - self._result_cap
        return text[: self._result_cap] + f"... [cortado: {cut} caracteres]"


class _AgentRun:
    """Uma execução do agente: liga modelo, histórico, ferramentas e consumo."""

    def __init__(
        self,
        model: ChatModelPort,
        conversation: Conversation,
        tools: ToolRunner,
        meter: TokenMeter,
        tracer: Tracer | None,
    ) -> None:
        self._model = model
        self._conversation = conversation
        self._tools = tools
        self.meter = meter
        self._tracer = tracer

    async def ask_model(self, turn: int, *, offer_tools: bool) -> AgentMessage:
        """Faz uma volta com o modelo, cobra os tokens e guarda a resposta no histórico."""
        sent = self._conversation.snapshot()
        started = time.perf_counter()
        reply = await self._model.complete(sent, self._tools.specs if offer_tools else [])
        tokens = self.meter.charge(sent, reply)
        self._trace_turn(turn, reply, tokens, started)
        self._conversation.add(reply)
        return reply

    async def execute(self, calls: Sequence[ToolCall]) -> None:
        """Executa cada pedido e devolve o resultado ao modelo pelo histórico."""
        for call in calls:
            self._conversation.add_tool_result(call, await self._tools.run(call))

    def finish(self, turns: int, reason: StopReason) -> AgentResult:
        log = self._tools.log
        return AgentResult(
            final_text=self._conversation.last_answer() + log.footer(),
            iterations=turns,
            tool_calls_executed=len(log.executed),
            stopped_by=reason,
            errors=log.errors,
            tokens_used=self.meter.used,
        )

    def _trace_turn(self, turn: int, reply: AgentMessage, tokens: int, started: float) -> None:
        if self._tracer is None:
            return
        self._tracer.event(
            "agent.turn",
            turn=turn,
            tool_calls=len(reply.tool_calls),
            tokens=tokens,
            ms=round((time.perf_counter() - started) * 1000),
        )
