"""Gerenciamento da conexão assíncrona com o banco (SQLAlchemy ``AsyncSession``).

Arquitetura BCE:
    * Control  -> ``DatabaseManager``: ciclo de vida do engine/pool, criação de
      tabelas, health-check e unidade de trabalho transacional.
    * Boundary -> ``get_database_manager()`` e ``get_db_session()``: pontos de
      integração com o FastAPI (lifespan e ``Depends``) no DIA 3.

Uso típico:
    manager = get_database_manager()
    await manager.create_tables()
    async with manager.session() as session:   # commit/rollback automáticos
        session.add(log)
    await manager.dispose()
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.config import DatabaseSettings, get_settings
from src.database.models import Base

logger = logging.getLogger(__name__)


# =============================================================================
# CONTROL
# =============================================================================
class DatabaseManager:
    """Encapsula engine + fábrica de sessões (uma instância por processo)."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings = settings
        self._engine: AsyncEngine = create_async_engine(
            settings.dsn, **self._engine_kwargs(settings)
        )
        self._session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,  # objetos continuam legíveis após o commit
            autoflush=False,
        )
        logger.info("Engine assíncrono criado para %s", settings.safe_dsn)

    @staticmethod
    def _engine_kwargs(settings: DatabaseSettings) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"echo": settings.echo, "pool_pre_ping": True}
        if not settings.is_sqlite:
            # SQLite (aiosqlite) não suporta parâmetros de QueuePool.
            kwargs |= {
                "pool_size": settings.pool_size,
                "max_overflow": settings.max_overflow,
                "pool_timeout": settings.pool_timeout_seconds,
                "pool_recycle": settings.pool_recycle_seconds,
            }
        return kwargs

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Unidade de trabalho: commit ao final, rollback em qualquer exceção."""
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def create_tables(self) -> None:
        """Cria as tabelas ausentes (idempotente). Em produção, prefira Alembic."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Tabelas verificadas/criadas: %s", sorted(Base.metadata.tables))

    async def ping(self) -> bool:
        """Health-check leve (``SELECT 1``); nunca propaga exceção."""
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001 — health-check deve ser tolerante
            logger.exception("Falha no health-check do banco de dados.")
            return False

    async def dispose(self) -> None:
        """Fecha todas as conexões do pool (chamar no shutdown)."""
        await self._engine.dispose()
        logger.info("Pool de conexões encerrado.")


# =============================================================================
# BOUNDARY
# =============================================================================
@lru_cache(maxsize=1)
def get_database_manager() -> DatabaseManager:
    """Instância única do ``DatabaseManager`` construída a partir de ``Settings``."""
    return DatabaseManager(get_settings().database)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Dependência FastAPI: uma sessão transacional por request."""
    async with get_database_manager().session() as session:
        yield session
