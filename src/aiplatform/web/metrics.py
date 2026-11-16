"""Métricas Prometheus da API, num registro próprio por aplicação.

Registro próprio (em vez do global) permite criar várias apps no mesmo
processo, como nos testes, sem colisão de nomes de métrica.
"""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)


class ApiMetrics:
    CONTENT_TYPE = CONTENT_TYPE_LATEST

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self._registry = registry or CollectorRegistry()
        self._requests = Counter(
            "http_requests_total", "Requisições", ["path", "status"], registry=self._registry
        )
        self._latency = Histogram(
            "http_request_seconds", "Latência", ["path"], registry=self._registry
        )
        self._agent_runs = Counter(
            "agent_runs_total", "Execuções do agente", ["stopped_by"], registry=self._registry
        )

    def observe_request(self, path: str, status: int, seconds: float) -> None:
        """`path` é o molde da rota (ex.: /documents), não a URL: mantém a cardinalidade baixa."""
        self._latency.labels(path).observe(seconds)
        self._requests.labels(path, str(status)).inc()

    def count_agent_run(self, stopped_by: str) -> None:
        self._agent_runs.labels(stopped_by).inc()

    def render(self) -> bytes:
        """Texto no formato de exposição do Prometheus, para a rota /metrics."""
        return generate_latest(self._registry)
