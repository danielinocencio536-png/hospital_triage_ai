from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipelines.evaluate_model import AGE_LABELS, calibration_report, evaluate, group_metrics
from src.pipelines.train_model import ModelBundle, TrainingConfig, split_holdout
from tests.conftest import FAST_TRAINING_CONFIG


def test_modelo_perfeitamente_calibrado_tem_ece_baixo() -> None:
    rng = np.random.default_rng(0)
    proba = rng.uniform(0, 1, 50_000)
    y = (rng.uniform(0, 1, proba.size) < proba).astype(np.int64)
    report = calibration_report(y, proba)
    assert report.expected_calibration_error < 0.02
    assert len(report.bins) == 10
    assert sum(b.count for b in report.bins) == proba.size


def test_superestimacao_aparece_no_ece() -> None:
    y = np.array([0, 0, 0, 0, 1] * 20, dtype=np.int64)
    proba = np.full(y.size, 0.8)
    report = calibration_report(y, proba)
    assert report.mean_predicted == pytest.approx(0.8)
    assert report.observed_rate == pytest.approx(0.2)
    assert report.expected_calibration_error == pytest.approx(0.6)
    assert report.brier_score > report.brier_reference


def test_metricas_por_grupo_de_matriz_conhecida() -> None:
    groups = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0], dtype=np.int64)
    proba = np.array([0.9, 0.2, 0.7, 0.1, 0.3, 0.8, 0.6, 0.1])
    a, b = group_metrics(groups, y, proba, threshold=0.5)
    assert (a.group, a.n, a.sensitivity, a.specificity, a.ppv) == ("A", 4, 0.5, 0.5, 0.5)
    assert (b.sensitivity, b.specificity, b.ppv) == (0.0, pytest.approx(1 / 3, abs=1e-4), 0.0)


def test_grupo_com_uma_classe_so_fica_sem_auc() -> None:
    groups = pd.Series(["A", "A", "B", "B"])
    y = np.array([1, 0, 0, 0], dtype=np.int64)
    a, b = group_metrics(groups, y, np.array([0.9, 0.1, 0.2, 0.3]), threshold=0.5)
    assert a.auc_roc == 1.0
    assert b.auc_roc is None


def test_split_bate_com_o_treino(synthetic_df: pd.DataFrame, model_bundle: ModelBundle) -> None:
    schema = model_bundle.schema
    X, y = synthetic_df[schema.all_features], synthetic_df[schema.target].to_numpy(dtype=np.int64)
    _, X_test, _, y_test = split_holdout(X, y, FAST_TRAINING_CONFIG)

    report = evaluate(model_bundle, X_test, y_test)

    # se o split fosse outro, o Brier não bateria com o que o treino gravou
    assert report["calibration"]["brier_score"] == model_bundle.metrics.brier_score
    assert report["n_test"] == len(y_test)
    assert {g["group"] for g in report["fairness"]["gender"]["groups"]} == {"F", "M"}
    bands = {g["group"] for g in report["fairness"]["age_band"]["groups"]}
    assert bands <= set(AGE_LABELS)
    assert sum(g["n"] for g in report["fairness"]["age_band"]["groups"]) == len(y_test)


def test_seed_diferente_muda_o_split(synthetic_df: pd.DataFrame) -> None:
    X = synthetic_df.drop(columns="readmitted_30d")
    y = synthetic_df["readmitted_30d"].to_numpy(dtype=np.int64)
    _, a, _, _ = split_holdout(X, y, TrainingConfig(seed=1))
    _, b, _, _ = split_holdout(X, y, TrainingConfig(seed=2))
    assert not a.index.equals(b.index)
