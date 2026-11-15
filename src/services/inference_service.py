"""Serviço de inferência do risco de readmissão em 30 dias.

Arquitetura BCE:
    * Entity   -> ``InferenceResult``: predição + latência (resultado do caso de uso).
    * Control  -> ``TriageInferenceService``: transforma o request em features,
      calcula a probabilidade, estratifica o risco e explica a predição com
      SHAP local agregado por feature clínica original.
    * Boundary -> ``ModelLoader`` (lê o ``ModelBundle`` do disco) e
      ``AuditLogRepository`` (grava ``triage_audit_logs``).

Decisões técnicas:
    * O ``shap.TreeExplainer`` é construído uma única vez no startup — criar o
      explainer por request custaria dezenas de milissegundos.
    * ``predict`` é CPU-bound e síncrono; o router o executa em threadpool para
      não bloquear o event loop. Um ``threading.Lock`` serializa o acesso ao
      pipeline/explainer, cuja segurança em múltiplas threads não é garantida.
    * Os valores SHAP das colunas one-hot (``gender_F``, ``gender_M``…) são
      somados de volta na feature original (``gender``), usando o
      ``feature_name_map`` salvo no treino — o clínico vê "gender", não dummies.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

from src.core.config import RiskStratificationSettings
from src.database.connection import DatabaseManager
from src.database.models import TriageAuditLog
from src.models.triage_schemas import (
    PatientTriageRequest,
    RiskCategory,
    RiskFactor,
    TriagePredictionResponse,
)
from src.pipelines.train_model import ModelBundle

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEÇÕES DE DOMÍNIO
# =============================================================================
class ModelLoadError(RuntimeError):
    """O artefato do modelo não existe ou é inválido."""


class InferenceError(RuntimeError):
    """Falha ao calcular a predição para um request válido."""


# =============================================================================
# ENTITY
# =============================================================================
@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Resposta da predição + latência medida (para auditoria/observabilidade)."""

    response: TriagePredictionResponse
    features: dict[str, Any]
    latency_ms: float


# =============================================================================
# BOUNDARY — carregamento do artefato
# =============================================================================
class ModelLoader:
    """Lê e valida o ``ModelBundle`` serializado pelo pipeline do DIA 1."""

    @staticmethod
    def load(path: Path) -> ModelBundle:
        if not path.is_file():
            raise ModelLoadError(
                f"Modelo não encontrado em {path}. "
                "Execute: python -m src.pipelines.train_model"
            )
        try:
            bundle = joblib.load(path)
        except Exception as exc:  # noqa: BLE001 — qualquer falha de unpickle é fatal
            raise ModelLoadError(f"Falha ao desserializar o modelo em {path}: {exc}") from exc
        if not isinstance(bundle, ModelBundle):
            raise ModelLoadError(
                f"Artefato inválido em {path}: esperado ModelBundle, recebido {type(bundle).__name__}."
            )
        logger.info(
            "Modelo v%s carregado (treinado em %s, limiar=%.4f, AUC holdout=%.4f).",
            bundle.model_version,
            bundle.trained_at,
            bundle.decision_threshold,
            bundle.metrics.auc_roc,
        )
        return bundle


# =============================================================================
# CONTROL — caso de uso de inferência
# =============================================================================
class TriageInferenceService:
    """Executa a predição de risco e a explicação SHAP para um paciente."""

    def __init__(self, bundle: ModelBundle, risk_settings: RiskStratificationSettings) -> None:
        self._bundle = bundle
        self._risk = risk_settings
        self._feature_order = bundle.schema.all_features
        self._preprocessor: ColumnTransformer = bundle.pipeline.named_steps["preprocessor"]
        self._classifier: XGBClassifier = bundle.pipeline.named_steps["classifier"]
        self._explainer = shap.TreeExplainer(self._classifier)
        self._lock = threading.Lock()

    @classmethod
    def from_path(cls, path: Path, risk_settings: RiskStratificationSettings) -> TriageInferenceService:
        return cls(ModelLoader.load(path), risk_settings)

    # --- Metadados expostos ao health-check ----------------------------------
    @property
    def model_version(self) -> str:
        return self._bundle.model_version

    @property
    def decision_threshold(self) -> float:
        return self._bundle.decision_threshold

    @property
    def trained_at(self) -> str:
        return self._bundle.trained_at

    # --- Caso de uso ---------------------------------------------------------
    def predict(self, request: PatientTriageRequest) -> InferenceResult:
        """Probabilidade, categoria de risco e top-K fatores (SHAP local)."""
        started = time.perf_counter()
        features = request.to_model_features()
        frame = self._to_frame(features)

        try:
            with self._lock:
                probability = float(self._bundle.pipeline.predict_proba(frame)[0, 1])
                shap_row = self._local_shap(frame)
        except Exception as exc:  # noqa: BLE001 — encapsula erros de libs de ML
            logger.exception("Falha na inferência do modelo.")
            raise InferenceError("Não foi possível calcular a predição.") from exc

        probability = float(np.clip(probability, 0.0, 1.0))
        response = TriagePredictionResponse(
            patient_id=request.patient_id,
            risk_probability=round(probability, 6),
            risk_category=RiskCategory.from_probability(
                probability, self._risk.moderate_threshold, self._risk.high_threshold
            ),
            readmission_predicted=probability >= self._bundle.decision_threshold,
            decision_threshold=round(self._bundle.decision_threshold, 6),
            top_risk_factors=self._top_factors(shap_row, features),
            model_version=self._bundle.model_version,
        )
        latency_ms = (time.perf_counter() - started) * 1_000
        return InferenceResult(response=response, features=features, latency_ms=round(latency_ms, 3))

    # --- Etapas internas -----------------------------------------------------
    def _to_frame(self, features: dict[str, Any]) -> pd.DataFrame:
        """DataFrame de 1 linha na ordem canônica do ``FeatureSchema``."""
        missing = set(self._feature_order) - features.keys()
        if missing:
            raise InferenceError(f"Features ausentes para o modelo: {sorted(missing)}")
        return pd.DataFrame([[features[name] for name in self._feature_order]], columns=self._feature_order)

    def _local_shap(self, frame: pd.DataFrame) -> dict[str, float]:
        """SHAP do paciente, somado por feature clínica original (log-odds)."""
        transformed = self._preprocessor.transform(frame)
        values = np.asarray(self._explainer.shap_values(transformed)).reshape(-1)
        names = self._bundle.transformed_feature_names
        if len(values) != len(names):
            raise InferenceError(
                f"SHAP retornou {len(values)} valores para {len(names)} features transformadas."
            )
        aggregated: dict[str, float] = {}
        for name, value in zip(names, values, strict=True):
            original = self._bundle.feature_name_map.get(name, name)
            aggregated[original] = aggregated.get(original, 0.0) + float(value)
        return aggregated

    def _top_factors(self, shap_row: dict[str, float], features: dict[str, Any]) -> list[RiskFactor]:
        ranked = sorted(shap_row.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return [
            RiskFactor.from_shap(feature=name, value=features[name], shap_value=value)
            for name, value in ranked[: self._risk.top_k_factors]
        ]


# =============================================================================
# BOUNDARY — persistência da auditoria
# =============================================================================
class AuditLogRepository:
    """Grava cada predição em ``triage_audit_logs`` (transação própria)."""

    def __init__(self, database: DatabaseManager) -> None:
        self._database = database

    async def save(self, result: InferenceResult) -> None:
        response = result.response
        log = TriageAuditLog(
            request_id=response.request_id,
            patient_id=response.patient_id,
            input_features=result.features,
            risk_probability=response.risk_probability,
            risk_category=response.risk_category.value,
            readmission_predicted=response.readmission_predicted,
            decision_threshold=response.decision_threshold,
            top_risk_factors=[factor.model_dump(mode="json") for factor in response.top_risk_factors],
            model_version=response.model_version,
            inference_latency_ms=result.latency_ms,
        )
        async with self._database.session() as session:
            session.add(log)
