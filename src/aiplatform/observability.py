"""Logs estruturados em JSON com o id da requisição."""

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from aiplatform.application.ports import Tracer

request_id: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    """Uma linha JSON por registro, sempre com o request_id."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id.get(),
        }
        attrs = getattr(record, "attrs", None)
        if attrs:
            entry["attrs"] = attrs
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def configure_logging() -> None:
    """Liga o log JSON no logger `aiplatform`. Chamar de novo não duplica o handler."""
    logger = logging.getLogger("aiplatform")
    if any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class LoggingTracer(Tracer):
    """Eventos de LLM como logs JSON; troque por Langfuse ou OpenTelemetry sem mudar o loop."""

    def __init__(self) -> None:
        self._log = logging.getLogger("aiplatform.trace")

    def event(self, name: str, **attrs: Any) -> None:
        self._log.info(name, extra={"attrs": attrs})
