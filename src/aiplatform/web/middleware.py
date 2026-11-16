"""Middleware de observabilidade e tradução de erros para HTTP."""

import logging
import re
import time
import uuid

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from aiplatform.observability import request_id
from aiplatform.web.metrics import ApiMetrics

log = logging.getLogger("aiplatform")

# O id vem do cliente e vai para os logs: só aceita um formato curto e seguro.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class RequestObserver:
    """Envolve cada requisição: id, métrica, log de acesso e erro 500 controlado.

    Toda requisição é contada com o status real, inclusive as que levantam
    exceção, que viram 500 com o request_id no corpo para o cliente reportar.
    """

    def __init__(self, metrics: ApiMetrics) -> None:
        self._metrics = metrics

    async def __call__(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        token = request_id.set(_request_id_from(request))
        started = time.perf_counter()
        try:
            response = await _call_or_internal_error(request, call_next)
            self._record(request, response.status_code, time.perf_counter() - started)
            response.headers["x-request-id"] = request_id.get()
            return response
        finally:
            request_id.reset(token)

    def _record(self, request: Request, status: int, seconds: float) -> None:
        self._metrics.observe_request(_route_template(request), status, seconds)
        log.info("%s %s -> %s", request.method, request.url.path, status)


async def provider_unavailable(_: Request, exc: Exception) -> Response:
    """Falha do provedor de IA é culpa de terceiro, não nossa: 502, não 500."""
    log.warning("provedor indisponível: %s", exc)
    return JSONResponse({"detail": "Provedor de IA indisponível."}, status_code=502)


async def _call_or_internal_error(request: Request, call_next: RequestResponseEndpoint) -> Response:
    try:
        return await call_next(request)
    except Exception:
        log.exception("erro não tratado em %s %s", request.method, request.url.path)
        body = {"detail": "Erro interno.", "request_id": request_id.get()}
        return JSONResponse(body, status_code=500)


def _request_id_from(request: Request) -> str:
    """Reaproveita o id do cliente se for seguro; senão gera um novo."""
    candidate = request.headers.get("x-request-id", "")
    return candidate if _SAFE_REQUEST_ID.match(candidate) else uuid.uuid4().hex[:12]


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    return route.path if route is not None else "unmatched"
