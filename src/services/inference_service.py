"""Inferência: probabilidade, categoria de risco e fatores SHAP de um paciente.

O TreeExplainer é criado uma vez só, porque montar um por request deixaria a
rota lenta. predict() é síncrono e o router roda em threadpool; o Lock está aí
porque não há garantia de que o pipeline e o explainer sejam thread-safe.

Os SHAP das colunas one-hot são somados de volta (gender_F + gender_M -> gender)
para a resposta mostrar a variável original.
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


class ModelLoadError(RuntimeError):
    """O artefato do modelo não existe ou é inválido."""


class InferenceError(RuntimeError):
    """Falha ao calcular a predição para um request válido."""


@dataclass(frozen=True, slots=True)
class InferenceResult:
    response: TriagePredictionResponse
    features: dict[str, Any]
    latency_ms: float


class ModelLoader:
    @staticmethod
    def load(path: Path) -> ModelBundle:
        if not path.is_file():
            raise ModelLoadError(
                f"Modelo não encontrado em {path}. "
                "Execute: python -m src.pipelines.train_model"
            )
        try:
            bundle = joblib.load(path)
        except Exception as exc:  # noqa: BLE001
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


class TriageInferenceService:
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

    @property
    def model_version(self) -> str:
        return self._bundle.model_version

    @property
    def decision_threshold(self) -> float:
        return self._bundle.decision_threshold

    @property
    def trained_at(self) -> str:
        return self._bundle.trained_at

    def predict(self, request: PatientTriageRequest) -> InferenceResult:
        started = time.perf_counter()
        features = request.to_model_features()
        frame = self._to_frame(features)

        try:
            with self._lock:
                probability = float(self._bundle.pipeline.predict_proba(frame)[0, 1])
                shap_row = self._local_shap(frame)
        except Exception as exc:  # noqa: BLE001
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

    def _to_frame(self, features: dict[str, Any]) -> pd.DataFrame:
        missing = set(self._feature_order) - features.keys()
        if missing:
            raise InferenceError(f"Features ausentes para o modelo: {sorted(missing)}")
        return pd.DataFrame([[features[name] for name in self._feature_order]], columns=self._feature_order)

    def _local_shap(self, frame: pd.DataFrame) -> dict[str, float]:
        """SHAP do paciente em log-odds, somado por feature original."""
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


class AuditLogRepository:
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
