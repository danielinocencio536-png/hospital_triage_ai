"""Endpoints HTTP da triagem de readmissão.

Arquitetura BCE — camada Boundary: valida a entrada (Pydantic), delega ao
``TriageInferenceService`` (Control) e grava a auditoria via
``AuditLogRepository``. Nenhuma regra de negócio vive aqui.

Rotas:
    POST /api/v1/triage/predict  -> risco de readmissão + top fatores SHAP
    GET  /health                 -> estado do modelo e do banco (sem prefixo,
                                    para probes de Docker/Kubernetes)

Dependências são resolvidas a partir de ``app.state`` (preenchido no lifespan
de ``src/main.py``) e podem ser substituídas nos testes com
``app.dependency_overrides``.
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from src.core.config import Settings, get_settings
from src.database.connection import DatabaseManager
from src.models.triage_schemas import PatientTriageRequest, TriagePredictionResponse
from src.services.inference_service import (
    AuditLogRepository,
    InferenceError,
    TriageInferenceService,
)

logger = logging.getLogger(__name__)

#: Header que informa ao cliente se a predição foi registrada na auditoria.
AUDIT_HEADER = "X-Audit-Logged"


# =============================================================================
# DEPENDÊNCIAS
# =============================================================================
def get_inference_service(request: Request) -> TriageInferenceService:
    service: TriageInferenceService | None = getattr(request.app.state, "inference_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo de predição indisponível. Verifique o artefato e o /health.",
        )
    return service


def get_audit_repository(request: Request) -> AuditLogRepository | None:
    return getattr(request.app.state, "audit_repository", None)


def get_database(request: Request) -> DatabaseManager | None:
    return getattr(request.app.state, "database", None)


InferenceServiceDep = Annotated[TriageInferenceService, Depends(get_inference_service)]
AuditRepositoryDep = Annotated[AuditLogRepository | None, Depends(get_audit_repository)]
DatabaseDep = Annotated[DatabaseManager | None, Depends(get_database)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


# =============================================================================
# TRIAGEM
# =============================================================================
triage_router = APIRouter(prefix="/triage", tags=["Triagem"])


@triage_router.post(
    "/predict",
    response_model=TriagePredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Prediz o risco de readmissão em 30 dias",
    responses={
        422: {"description": "Dados clínicos inválidos."},
        500: {"description": "Falha interna na inferência."},
        503: {"description": "Modelo não carregado."},
    },
)
async def predict_readmission_risk(
    payload: PatientTriageRequest,
    response: Response,
    service: InferenceServiceDep,
    audit: AuditRepositoryDep,
) -> TriagePredictionResponse:
    """Calcula probabilidade, categoria (Low/Moderate/High) e os fatores que mais
    pesaram na predição deste paciente.

    A auditoria é *fail-open*: se o banco estiver fora, a predição ainda é
    devolvida (o atendimento não para), mas o header ``X-Audit-Logged: false``
    sinaliza a falha e o erro é registrado no log da aplicação.
    """
    try:
        result = await run_in_threadpool(service.predict, payload)
    except InferenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    audited = False
    if audit is not None:
        try:
            await audit.save(result)
            audited = True
        except Exception:  # noqa: BLE001 — auditoria não pode derrubar a predição
            logger.exception(
                "Falha ao gravar auditoria do request_id=%s", result.response.request_id
            )
    response.headers[AUDIT_HEADER] = str(audited).lower()

    logger.info(
        "Predição request_id=%s categoria=%s p=%.4f latência=%.1fms auditado=%s",
        result.response.request_id,
        result.response.risk_category.value,
        result.response.risk_probability,
        result.latency_ms,
        audited,
    )
    return result.response


# =============================================================================
# HEALTH-CHECK
# =============================================================================
class ComponentStatus(BaseModel):
    model_loaded: bool
    model_version: str | None = None
    model_trained_at: str | None = None
    database: Literal["ok", "unavailable", "disabled"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    app_version: str
    environment: str
    components: ComponentStatus


health_router = APIRouter(tags=["Saúde"])


@health_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado do modelo e do banco",
    responses={503: {"model": HealthResponse, "description": "Modelo não carregado."}},
)
async def health(
    request: Request, response: Response, settings: SettingsDep, database: DatabaseDep
) -> HealthResponse:
    """``ok`` = modelo e banco ok · ``degraded`` = predições funcionam, mas sem
    auditoria · ``unavailable`` (HTTP 503) = sem modelo, a API não serve."""
    service: TriageInferenceService | None = getattr(request.app.state, "inference_service", None)

    if database is None:
        db_status: Literal["ok", "unavailable", "disabled"] = "disabled"
    else:
        db_status = "ok" if await database.ping() else "unavailable"

    if service is None:
        overall: Literal["ok", "degraded", "unavailable"] = "unavailable"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif db_status != "ok":
        overall = "degraded"
    else:
        overall = "ok"

    return HealthResponse(
        status=overall,
        app_version=settings.app_version,
        environment=settings.environment.value,
        components=ComponentStatus(
            model_loaded=service is not None,
            model_version=service.model_version if service else None,
            model_trained_at=service.trained_at if service else None,
            database=db_status,
        ),
    )
