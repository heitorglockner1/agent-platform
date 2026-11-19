"""Modelos de domínio: dataclasses puras, sem dependência de framework."""

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
StopReason = Literal["concluded", "max_iterations", "budget"]
ToolHandler = Callable[..., Awaitable[Any]]

_TENANT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,61}[a-z0-9]$")


@dataclass(frozen=True)
class TenantId:
    """Cliente dono dos dados. Todo acesso a documentos passa por um.

    O formato é restrito porque o id vira parte do nome da coleção no banco
    vetorial: minúsculas, dígitos, `-` e `_`, de 3 a 63 caracteres.
    """

    value: str

    def __post_init__(self) -> None:
        if not _TENANT_ID.match(self.value):
            raise ValueError(f"id de cliente inválido: {self.value!r}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Document:
    """Texto a indexar. `source` é o que aparece na citação (ex.: nome do arquivo)."""

    id: str
    text: str
    source: str = ""


@dataclass(frozen=True)
class Chunk:
    """Trecho de um documento; a unidade que é indexada e recuperada."""

    document_id: str
    index: int
    text: str
    source: str = ""

    @property
    def id(self) -> str:
        return f"{self.document_id}#{self.index}"


@dataclass(frozen=True)
class ScoredChunk:
    """Trecho recuperado com a similaridade da busca (1.0 é idêntico)."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Citation:
    """Fonte de uma resposta, montada pelo código a partir da recuperação."""

    chunk_id: str
    source: str
    score: float


@dataclass(frozen=True)
class Answer:
    """Resposta do RAG. `grounded` é False quando nada passou do limiar."""

    text: str
    citations: tuple[Citation, ...]
    grounded: bool


@dataclass(frozen=True)
class ToolSpec:
    """Ferramenta oferecida ao modelo. `parameters` é JSON Schema.

    O modelo nunca executa nada: só pede. Quem executa é o `handler`, chamado
    pelo loop, e é nele que moram as regras de negócio (nunca no prompt).
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler


@dataclass(frozen=True)
class ToolCall:
    """Pedido do modelo para executar uma ferramenta com estes argumentos."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class TokenUsage:
    """Tokens consumidos numa chamada ao modelo, como o provedor informou."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class AgentMessage:
    """Uma mensagem do histórico, em formato neutro de provedor."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    usage: TokenUsage | None = None  # informado pelo provedor, quando houver


@dataclass
class AgentResult:
    """Desfecho de uma execução do agente, com o que de fato foi executado."""

    final_text: str
    iterations: int
    tool_calls_executed: int
    stopped_by: StopReason
    errors: list[str] = field(default_factory=list)
    tokens_used: int = 0
