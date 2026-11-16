"""Orçamento de tokens por execução do agente."""

from aiplatform.application.ports import BudgetPort


class TokenBudget(BudgetPort):
    """Permite chamadas enquanto sobrar saldo; a última pode estourar o teto."""

    def __init__(self, max_tokens: int) -> None:
        self._left = max_tokens

    def allows_next_call(self) -> bool:
        return self._left > 0

    def record(self, tokens: int) -> None:
        self._left -= tokens
