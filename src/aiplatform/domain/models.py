"""Modelos de domínio: dataclasses puras, sem dependência de framework."""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source: str = ""


@dataclass(frozen=True)
class Chunk:
    document_id: str
    index: int
    text: str
    source: str = ""

    @property
    def id(self) -> str:
        return f"{self.document_id}#{self.index}"


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    source: str
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    citations: List[Citation]
    grounded: bool  # False quando nada relevante foi recuperado


@dataclass(frozen=True)
class ToolSpec:
    """Ferramenta oferecida ao modelo. `parameters` é JSON Schema.

    O modelo nunca executa nada: só pede. Quem executa é o `handler`, chamado
    pelo loop, e é nele que moram as regras de negócio (nunca no prompt).
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class AgentMessage:
    role: str  # system | user | assistant | tool
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class AgentResult:
    final_text: str
    iterations: int
    tool_calls_executed: int
    stopped_by: str  # concluded | max_iterations | budget
    errors: List[str] = field(default_factory=list)
