"""Ponto de entrada da API FastAPI.

Arquitetura BCE — Boundary de mais alto nível: monta a aplicação, conecta
middlewares e routers e controla o ciclo de vida dos recursos (lifespan).

Startup:
    1. Carrega o ``ModelBundle`` e constrói o ``TriageInferenceService``
       (SHAP explainer incluso) uma única vez por processo.
    2. Cria o ``DatabaseManager`` e, se configurado, as tabelas ausentes.
Shutdown:
    3. Fecha o pool de conexões.

Falhas no startup não derrubam o processo: a API sobe em modo degradado e o
``/health`` reporta o problema (503 sem modelo; "degraded" sem banco). Em
produção, o probe do orquestrador decide se reinicia o container.

Execução:
    uvicorn src.main:app --reload            # desenvolvimento
    python -m src.main                       # atalho equivalente
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v1.triage_router import health_router, triage_router
from src.core.config import Settings, get_settings
from src.database.connection import get_database_manager
from src.services.inference_service import (
    AuditLogRepository,
    ModelLoadError,
    TriageInferenceService,
)

logger = logging.getLogger("src.main")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    # --- Modelo --------------------------------------------------------------
    app.state.inference_service = None
    try:
        app.state.inference_service = TriageInferenceService.from_path(
            settings.model_path, settings.risk
        )
    except ModelLoadError:
        logger.exception("API iniciada SEM modelo — /triage/predict responderá 503.")

    # --- Banco de dados ------------------------------------------------------
    database = get_database_manager()
    app.state.database = database
    app.state.audit_repository = AuditLogRepository(database)
    if settings.database.create_tables_on_startup:
        try:
            await database.create_tables()
        except Exception:  # noqa: BLE001 — banco fora não impede predições
            logger.exception("Banco indisponível no startup (%s).", settings.database.safe_dsn)

    logger.info("%s v%s pronta (%s).", settings.app_name, settings.app_version, settings.environment)
    try:
        yield
    finally:
        await database.dispose()
        logger.info("Recursos liberados — shutdown concluído.")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Fábrica da aplicação (permite instâncias isoladas nos testes)."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Triagem de risco de readmissão hospitalar em 30 dias com XGBoost "
            "e explicabilidade SHAP. Ferramenta de apoio à decisão clínica — "
            "não substitui o julgamento profissional."
        ),
        lifespan=lifespan,
        debug=settings.debug,
        # Swagger desabilitado em produção para não expor a superfície da API.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None if settings.is_production else "/redoc",
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
        expose_headers=["X-Audit-Logged"],
    )

    app.include_router(triage_router, prefix=settings.api_v1_prefix)
    app.include_router(health_router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=False)
