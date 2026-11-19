"""Rotas: POST /api/v1/triage/predict e GET /health (sem prefixo, para os probes)."""

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

# diz ao cliente se a predição entrou na auditoria
AUDIT_HEADER = "X-Audit-Logged"


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
    """Retorna a probabilidade, a categoria (Low/Moderate/High) e os fatores que mais pesaram.

    Se a gravação da auditoria falhar, a predição volta mesmo assim, com X-Audit-Logged: false.
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
        except Exception:  # noqa: BLE001
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
    """ok, degraded (sem banco, logo sem auditoria) ou unavailable (503, sem modelo)."""
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
