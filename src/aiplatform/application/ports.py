"""Portas (interfaces) da aplicação. A infraestrutura as implementa.

Separar `ChatModelPort` (uma volta de conversa) do loop do agente deixa o loop
nosso, legível e testável com um fake pequeno; o que depende do formato do
provedor (OpenAI, Bedrock, LangChain) fica só no adaptador.

Regra: porta que faz I/O de rede é assíncrona (chat, embeddings, banco vetorial,
mensageria). `BudgetPort` e `Tracer` são locais e continuam síncronas.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from aiplatform.domain.models import AgentMessage, Chunk, ScoredChunk, TenantId, ToolSpec


class ProviderUnavailableError(Exception):
    """Falha do provedor externo (rede, timeout, limite). A API devolve 502."""


class ChatModelPort(ABC):
    @abstractmethod
    async def complete(
        self, messages: Sequence[AgentMessage], tools: Sequence[ToolSpec]
    ) -> AgentMessage:
        """Uma volta. Com `tools` vazio o modelo é obrigado a responder em texto.

        Falha do provedor levanta `ProviderUnavailableError`.
        """


class EmbedderPort(ABC):
    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Um vetor por texto. Falha do provedor levanta `ProviderUnavailableError`."""


class VectorStorePort(ABC):
    """Banco vetorial isolado por cliente. A similaridade vai de 0 a 1 (1 é idêntico).

    `tenant` é obrigatório em todo método, e não um filtro opcional: assim não
    existe chamada que esqueça o cliente. Um cliente nunca lê, sobrescreve nem
    apaga trechos de outro, mesmo com o mesmo id de documento.
    """

    @abstractmethod
    async def upsert(
        self, tenant: TenantId, chunks: Sequence[Chunk], vectors: Sequence[list[float]]
    ) -> None:
        """Grava ou substitui cada trecho do cliente pelo seu `Chunk.id`."""

    @abstractmethod
    async def search(self, tenant: TenantId, vector: list[float], top_k: int) -> list[ScoredChunk]:
        """Os `top_k` trechos do cliente mais parecidos, do mais para o menos parecido."""

    @abstractmethod
    async def delete_document(self, tenant: TenantId, document_id: str) -> None:
        """Remove todos os trechos do documento do cliente."""


class EventPublisherPort(ABC):
    """Mensageria. Adaptadores possíveis: em memória, SQS, Kafka, RabbitMQ."""

    @abstractmethod
    async def publish(self, topic: str, payload: dict[str, Any]) -> None: ...


class BudgetPort(ABC):
    """Teto de custo de uma execução, consultado antes de cada chamada ao modelo."""

    @abstractmethod
    def allows_next_call(self) -> bool: ...

    @abstractmethod
    def record(self, tokens: int) -> None: ...


class Tracer(ABC):
    """Observabilidade de chamadas de LLM (Langfuse, OpenTelemetry...)."""

    @abstractmethod
    def event(self, name: str, **attrs: Any) -> None: ...
