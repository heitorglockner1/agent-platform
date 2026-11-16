"""Portas (interfaces) da aplicação. A infraestrutura as implementa.

Separar `ChatModelPort` (uma volta de conversa) do loop do agente deixa o loop
nosso, legível e testável com um fake pequeno; o que depende do formato do
provedor (OpenAI, Bedrock, LangChain) fica só no adaptador.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from aiplatform.domain.models import AgentMessage, Chunk, ScoredChunk, ToolSpec


class ChatModelPort(ABC):
    @abstractmethod
    async def complete(
        self, messages: Sequence[AgentMessage], tools: Sequence[ToolSpec]
    ) -> AgentMessage:
        """Uma volta. Com `tools` vazio o modelo é obrigado a responder em texto."""


class EmbedderPort(ABC):
    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class VectorStorePort(ABC):
    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None: ...

    @abstractmethod
    def search(self, vector: list[float], top_k: int) -> list[ScoredChunk]: ...

    @abstractmethod
    def delete_document(self, document_id: str) -> None: ...


class EventPublisherPort(ABC):
    """Mensageria. Adaptadores possíveis: em memória, SQS, Kafka, RabbitMQ."""

    @abstractmethod
    def publish(self, topic: str, payload: dict[str, Any]) -> None: ...


class BudgetPort(ABC):
    @abstractmethod
    def allows_next_call(self) -> bool: ...

    @abstractmethod
    def record(self, tokens: int) -> None: ...


class Tracer(ABC):
    """Observabilidade de chamadas de LLM (Langfuse, OpenTelemetry...)."""

    @abstractmethod
    def event(self, name: str, **attrs: Any) -> None: ...
