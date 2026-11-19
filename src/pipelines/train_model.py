"""Treino do modelo de readmissão em 30 dias (XGBoost + SHAP).

Algumas escolhas:
- desbalanceamento tratado com scale_pos_weight (n_neg / n_pos), sem SMOTE;
- o limiar de decisão sai do índice de Youden em predições out-of-fold do treino,
  então o conjunto de teste só é usado na avaliação final;
- RobustScaler nas numéricas porque o tempo de internação tem cauda longa.

    python -m src.pipelines.train_model --data data/raw/mimic_synthetic.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier

from src.pipelines.generate_mimic_data import COMORBIDITY_PREFIX, TARGET_COLUMN

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

MODEL_VERSION: Final[str] = "1.0.0"
DEFAULT_DATA_PATH: Final[Path] = Path("data/raw/mimic_synthetic.csv")
DEFAULT_MODELS_DIR: Final[Path] = Path("models")
BUNDLE_FILENAME: Final[str] = "readmission_model.joblib"
METRICS_FILENAME: Final[str] = "metrics_report.json"
SHAP_FILENAME: Final[str] = "shap_global_importance.json"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    test_size: float = 0.20
    cv_folds: int = 5
    seed: int = 42
    shap_sample_size: int = 1_000
    n_estimators: int = 400
    learning_rate: float = 0.04
    max_depth: int = 4
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    min_child_weight: float = 3.0
    reg_lambda: float = 1.0


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    categorical: tuple[str, ...]
    numerical: tuple[str, ...]
    binary: tuple[str, ...]
    target: str = TARGET_COLUMN
    identifier: str = "patient_id"

    @property
    def all_features(self) -> list[str]:
        return [*self.categorical, *self.numerical, *self.binary]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> FeatureSchema:
        binary = tuple(sorted(c for c in df.columns if c.startswith(COMORBIDITY_PREFIX)))
        if not binary:
            raise ValueError(f"Nenhuma coluna com prefixo '{COMORBIDITY_PREFIX}' encontrada.")
        schema = cls(
            categorical=("gender", "admission_type"),
            numerical=("age", "length_of_stay", "num_lab_procedures", "num_medications"),
            binary=binary,
        )
        missing = set(schema.all_features + [schema.target]) - set(df.columns)
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes no dataset: {sorted(missing)}")
        return schema


@dataclass(frozen=True, slots=True)
class ClinicalMetrics:
    threshold: float
    sensitivity_recall: float
    specificity: float
    ppv_precision: float
    npv: float
    f1_score: float
    auc_roc: float
    auc_pr: float
    brier_score: float
    prevalence: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


@dataclass(slots=True)
class ModelBundle:
    """Tudo que a API precisa para inferir: pipeline, esquema, limiar e métricas."""

    pipeline: Pipeline
    schema: FeatureSchema
    decision_threshold: float
    transformed_feature_names: list[str]
    feature_name_map: dict[str, str]
    metrics: ClinicalMetrics
    model_version: str = MODEL_VERSION
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    library_versions: dict[str, str] = field(
        default_factory=lambda: {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "shap": shap.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }
    )


class PipelineFactory:
    def __init__(self, config: TrainingConfig) -> None:
        self._config = config

    def build(self, schema: FeatureSchema, scale_pos_weight: float) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64),
                    list(schema.categorical),
                ),
                ("numerical", RobustScaler(), list(schema.numerical)),
                ("binary", "passthrough", list(schema.binary)),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        cfg = self._config
        classifier = XGBClassifier(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            min_child_weight=cfg.min_child_weight,
            reg_lambda=cfg.reg_lambda,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=cfg.seed,
            n_jobs=-1,
        )
        return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


class ThresholdSelector:
    @staticmethod
    def youden(y_true: IntArray, y_proba: FloatArray) -> float:
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        j_scores = tpr - fpr
        best = int(np.argmax(j_scores))
        # roc_curve pode retornar +inf no primeiro limiar; limita a [0, 1].
        return float(np.clip(thresholds[best], 0.0, 1.0))


class ClinicalEvaluator:
    @staticmethod
    def evaluate(y_true: IntArray, y_proba: FloatArray, threshold: float) -> ClinicalMetrics:
        y_pred = (y_proba >= threshold).astype(np.int64)
        tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())

        def _safe_div(num: int, den: int) -> float:
            return float(num / den) if den else 0.0

        return ClinicalMetrics(
            threshold=round(threshold, 4),
            sensitivity_recall=round(_safe_div(tp, tp + fn), 4),
            specificity=round(_safe_div(tn, tn + fp), 4),
            ppv_precision=round(_safe_div(tp, tp + fp), 4),
            npv=round(_safe_div(tn, tn + fn), 4),
            f1_score=round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            auc_roc=round(float(roc_auc_score(y_true, y_proba)), 4),
            auc_pr=round(float(average_precision_score(y_true, y_proba)), 4),
            brier_score=round(float(brier_score_loss(y_true, y_proba)), 4),
            prevalence=round(float(np.mean(y_true)), 4),
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
        )


class ShapExplainer:
    def __init__(self, pipeline: Pipeline, feature_name_map: dict[str, str]) -> None:
        self._preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
        self._classifier: XGBClassifier = pipeline.named_steps["classifier"]
        self._feature_name_map = feature_name_map

    def global_importance(self, X: pd.DataFrame) -> dict[str, dict[str, float]]:
        """mean(|SHAP|) por coluna transformada e somado por feature original."""
        X_transformed = self._preprocessor.transform(X)
        names = list(self._preprocessor.get_feature_names_out())
        explainer = shap.TreeExplainer(self._classifier)
        shap_values = np.asarray(explainer.shap_values(X_transformed))
        mean_abs = np.abs(shap_values).mean(axis=0)

        per_transformed = dict(zip(names, (float(v) for v in mean_abs), strict=True))
        per_original: dict[str, float] = {}
        for name, value in per_transformed.items():
            original = self._feature_name_map[name]
            per_original[original] = per_original.get(original, 0.0) + value

        return {
            "transformed": _sort_desc(per_transformed),
            "original": _sort_desc(per_original),
        }


class ReadmissionModelTrainer:
    def __init__(
        self,
        config: TrainingConfig,
        pipeline_factory: PipelineFactory | None = None,
        evaluator: ClinicalEvaluator | None = None,
        threshold_selector: ThresholdSelector | None = None,
    ) -> None:
        self._config = config
        self._factory = pipeline_factory or PipelineFactory(config)
        self._evaluator = evaluator or ClinicalEvaluator()
        self._threshold_selector = threshold_selector or ThresholdSelector()

    def train(self, df: pd.DataFrame) -> tuple[ModelBundle, dict[str, dict[str, float]]]:
        cfg = self._config
        schema = FeatureSchema.from_dataframe(df)
        X = df[schema.all_features]
        y = df[schema.target].to_numpy(dtype=np.int64)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed
        )
        n_pos = int(y_train.sum())
        scale_pos_weight = (len(y_train) - n_pos) / max(n_pos, 1)
        logger.info(
            "Treino=%d | Teste=%d | Positivos treino=%.2f%% | scale_pos_weight=%.3f",
            len(y_train), len(y_test), 100 * n_pos / len(y_train), scale_pos_weight,
        )

        # limiar escolhido só com o treino (out-of-fold)
        cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
        oof_proba = cross_val_predict(
            self._factory.build(schema, scale_pos_weight),
            X_train, y_train, cv=cv, method="predict_proba",
        )[:, 1]
        threshold = self._threshold_selector.youden(y_train, oof_proba)
        oof_auc = roc_auc_score(y_train, oof_proba)
        logger.info("AUC-ROC out-of-fold (treino): %.4f | Limiar Youden: %.4f", oof_auc, threshold)

        pipeline = self._factory.build(schema, scale_pos_weight)
        pipeline.fit(X_train, y_train)

        test_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = self._evaluator.evaluate(y_test, test_proba, threshold)

        preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
        transformed_names = [str(n) for n in preprocessor.get_feature_names_out()]
        name_map = _build_feature_name_map(transformed_names, schema)
        sample = X_test.sample(
            n=min(cfg.shap_sample_size, len(X_test)), random_state=cfg.seed
        )
        shap_importance = ShapExplainer(pipeline, name_map).global_importance(sample)

        bundle = ModelBundle(
            pipeline=pipeline,
            schema=schema,
            decision_threshold=threshold,
            transformed_feature_names=transformed_names,
            feature_name_map=name_map,
            metrics=metrics,
        )
        return bundle, shap_importance


def _build_feature_name_map(transformed: list[str], schema: FeatureSchema) -> dict[str, str]:
    """gender_F -> gender; as outras colunas ficam como estão."""
    mapping: dict[str, str] = {}
    for name in transformed:
        original = next(
            (cat for cat in schema.categorical if name.startswith(f"{cat}_")), name
        )
        mapping[name] = original
    return mapping


def _sort_desc(values: dict[str, float]) -> dict[str, float]:
    return {k: round(v, 6) for k, v in sorted(values.items(), key=lambda kv: kv[1], reverse=True)}


class CsvDatasetLoader:
    @staticmethod
    def load(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset não encontrado em '{path}'. "
                "Execute antes: python -m src.pipelines.generate_mimic_data"
            )
        df = pd.read_csv(path, dtype={"patient_id": str})
        logger.info("Dataset carregado: %s (%d linhas)", path, len(df))
        return df


class ArtifactRepository:
    def __init__(self, models_dir: Path) -> None:
        self._dir = models_dir

    def save(
        self, bundle: ModelBundle, shap_importance: dict[str, dict[str, float]]
    ) -> dict[str, Path]:
        self._dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "bundle": self._dir / BUNDLE_FILENAME,
            "metrics": self._dir / METRICS_FILENAME,
            "shap": self._dir / SHAP_FILENAME,
        }
        joblib.dump(bundle, paths["bundle"], compress=3)

        report: dict[str, Any] = {
            "model_version": bundle.model_version,
            "trained_at": bundle.trained_at,
            "decision_threshold": bundle.decision_threshold,
            "holdout_metrics": asdict(bundle.metrics),
            "library_versions": bundle.library_versions,
        }
        paths["metrics"].write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
        paths["shap"].write_text(json.dumps(shap_importance, indent=2, ensure_ascii=False), "utf-8")

        for kind, path in paths.items():
            logger.info("Artefato '%s' salvo em %s", kind, path)
        return paths


def _print_report(metrics: ClinicalMetrics, shap_importance: dict[str, dict[str, float]]) -> None:
    rows = [
        ("Limiar de decisão (Youden)", metrics.threshold),
        ("Sensibilidade / Recall", metrics.sensitivity_recall),
        ("Especificidade", metrics.specificity),
        ("VPP / Precisão", metrics.ppv_precision),
        ("VPN", metrics.npv),
        ("F1-Score", metrics.f1_score),
        ("AUC-ROC", metrics.auc_roc),
        ("AUC-PR", metrics.auc_pr),
        ("Brier Score", metrics.brier_score),
        ("Prevalência (teste)", metrics.prevalence),
    ]
    print("\n" + "=" * 52)
    print("  RELATÓRIO DE MÉTRICAS CLÍNICAS (holdout)")
    print("=" * 52)
    for label, value in rows:
        print(f"  {label:<32} {value:>10.4f}")
    print("-" * 52)
    print(f"  Matriz de confusão: TP={metrics.true_positives} FP={metrics.false_positives} "
          f"TN={metrics.true_negatives} FN={metrics.false_negatives}")
    print("-" * 52)
    print("  Top 8 fatores de risco globais (mean |SHAP|):")
    for name, value in list(shap_importance["original"].items())[:8]:
        print(f"    {name:<38} {value:.4f}")
    print("=" * 52 + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o modelo de readmissão em 30 dias.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()
    df = CsvDatasetLoader.load(args.data)
    bundle, shap_importance = ReadmissionModelTrainer(TrainingConfig(seed=args.seed)).train(df)
    ArtifactRepository(args.models_dir).save(bundle, shap_importance)
    _print_report(bundle.metrics, shap_importance)


if __name__ == "__main__":
    # importa de novo pelo pacote para o pickle gravar src.pipelines.train_model.ModelBundle
    # e não __main__.ModelBundle (senão a API não consegue carregar o arquivo)
    from src.pipelines.train_model import main as _canonical_main

    _canonical_main()
