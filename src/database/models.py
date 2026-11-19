"""Tabela de auditoria: uma linha por predição feita pela API.

Só guarda o patient_id pseudonimizado e as features do request (LGPD).
Usa JSONB no Postgres e JSON no SQLite.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    Index,
    MetaData,
    String,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# nomes fixos de constraints, para as migrações do Alembic não mudarem à toa
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

PortableJSON = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TriageAuditLog(Base):
    __tablename__ = "triage_audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    request_id: Mapped[uuid.UUID] = mapped_column(Uuid, unique=True, nullable=False)
    patient_id: Mapped[str | None] = mapped_column(String(64), index=True)

    input_features: Mapped[dict[str, Any]] = mapped_column(PortableJSON, nullable=False)

    risk_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_category: Mapped[str] = mapped_column(String(16), nullable=False)
    readmission_predicted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    decision_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    top_risk_factors: Mapped[list[dict[str, Any]]] = mapped_column(
        PortableJSON, nullable=False, default=list
    )

    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    inference_latency_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "risk_probability >= 0 AND risk_probability <= 1", name="risk_probability_range"
        ),
        CheckConstraint(
            "risk_category IN ('Low', 'Moderate', 'High')", name="risk_category_valid"
        ),
        CheckConstraint(
            "decision_threshold >= 0 AND decision_threshold <= 1", name="decision_threshold_range"
        ),
        # ex.: "pacientes High nas últimas 24h"
        Index("ix_triage_audit_logs_category_created", "risk_category", "created_at"),
        Index("ix_triage_audit_logs_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"TriageAuditLog(request_id={self.request_id}, category={self.risk_category}, "
            f"p={self.risk_probability:.4f}, model={self.model_version})"
        )
