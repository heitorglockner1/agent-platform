"""Autenticação por API key: cada chave pertence a um cliente.

A chave vem no cabeçalho `Authorization: Bearer <chave>` e o cliente sai dela,
nunca do corpo da requisição: quem chama não escolhe de quem são os dados.

Sem chaves configuradas a API roda em modo aberto (demo local): toda
requisição pertence ao cliente `default`. Em produção, configure `API_KEYS`.
"""

import hashlib
import logging
from collections.abc import Mapping

from fastapi import Header, HTTPException

from aiplatform.domain.models import TenantId

log = logging.getLogger("aiplatform")

DEFAULT_TENANT = TenantId("default")


class TenantResolver:
    """Dependência do FastAPI que devolve o cliente dono da requisição."""

    def __init__(self, api_keys: Mapping[str, TenantId]) -> None:
        # Guarda só o hash das chaves: a busca não compara o segredo caractere a
        # caractere (sem vazamento por tempo) e um dump de memória não as expõe.
        self._tenants = {_digest(key): tenant for key, tenant in api_keys.items()}
        if not self._tenants:
            log.warning(
                "API sem autenticação: toda requisição usa o cliente %r", DEFAULT_TENANT.value
            )

    async def __call__(self, authorization: str | None = Header(default=None)) -> TenantId:
        if not self._tenants:
            return DEFAULT_TENANT
        tenant = self._tenants.get(_digest(_bearer_token(authorization)))
        if tenant is None:
            raise HTTPException(401, "API key ausente ou inválida.", {"WWW-Authenticate": "Bearer"})
        return tenant


def _bearer_token(authorization: str | None) -> str:
    scheme, _, token = (authorization or "").partition(" ")
    return token.strip() if scheme.lower() == "bearer" else ""


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()
