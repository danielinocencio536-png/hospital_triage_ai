"""Calibração e fairness do modelo treinado, no mesmo conjunto de teste do treino.

    python -m src.pipelines.evaluate_model --data data/raw/mimic_synthetic.csv

O split é refeito com a mesma semente e o mesmo test_size do train_model, então
o modelo é avaliado só em pacientes que ele não viu.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.pipelines.train_model import (
    BUNDLE_FILENAME,
    DEFAULT_DATA_PATH,
    DEFAULT_MODELS_DIR,
    CsvDatasetLoader,
    FloatArray,
    IntArray,
    ModelBundle,
    TrainingConfig,
    split_holdout,
)
from src.services.inference_service import ModelLoader

logger = logging.getLogger(__name__)

EVALUATION_FILENAME: Final[str] = "evaluation_report.json"
AGE_BINS: Final[list[float]] = [0, 44, 64, 79, np.inf]
AGE_LABELS: Final[list[str]] = ["18-44", "45-64", "65-79", "80+"]


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    mean_predicted: float
    observed_rate: float
    count: int


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    brier_score: float
    # Brier de quem chuta sempre a prevalência; um modelo útil precisa ficar abaixo disso
    brier_reference: float
    expected_calibration_error: float
    mean_predicted: float
    observed_rate: float
    bins: list[CalibrationBin]


@dataclass(frozen=True, slots=True)
class GroupMetrics:
    group: str
    n: int
    prevalence: float
    mean_predicted: float
    auc_roc: float | None
    sensitivity: float
    specificity: float
    ppv: float
    brier_score: float


def calibration_report(y_true: IntArray, y_proba: FloatArray, n_bins: int = 10) -> CalibrationReport:
    # bins por quantil: com prevalência de 15%, bins de largura fixa ficariam quase vazios no topo
    bin_ids = pd.qcut(y_proba, q=n_bins, labels=False, duplicates="drop")
    if np.isnan(bin_ids).all():  # probabilidade constante: qcut não consegue cortar
        bin_ids = np.zeros(len(y_proba))
    bins: list[CalibrationBin] = []
    ece = 0.0
    for b in np.unique(bin_ids):
        mask = bin_ids == b
        pred, obs = float(y_proba[mask].mean()), float(y_true[mask].mean())
        bins.append(CalibrationBin(round(pred, 4), round(obs, 4), int(mask.sum())))
        ece += mask.mean() * abs(pred - obs)

    prevalence = float(y_true.mean())
    return CalibrationReport(
        brier_score=round(float(brier_score_loss(y_true, y_proba)), 4),
        brier_reference=round(prevalence * (1 - prevalence), 4),
        expected_calibration_error=round(float(ece), 4),
        mean_predicted=round(float(y_proba.mean()), 4),
        observed_rate=round(prevalence, 4),
        bins=bins,
    )


def group_metrics(
    groups: pd.Series, y_true: IntArray, y_proba: FloatArray, threshold: float
) -> list[GroupMetrics]:
    y_pred = y_proba >= threshold
    result: list[GroupMetrics] = []
    for name in sorted(groups.dropna().unique()):
        mask = (groups == name).to_numpy()
        yt, yp, pr = y_true[mask], y_pred[mask], y_proba[mask]
        tp = int((yp & (yt == 1)).sum())
        fp = int((yp & (yt == 0)).sum())
        pos, neg = int((yt == 1).sum()), int((yt == 0).sum())
        result.append(
            GroupMetrics(
                group=str(name),
                n=int(mask.sum()),
                prevalence=round(float(yt.mean()), 4),
                mean_predicted=round(float(pr.mean()), 4),
                # AUC não existe num grupo com uma classe só
                auc_roc=round(float(roc_auc_score(yt, pr)), 4) if pos and neg else None,
                sensitivity=round(tp / pos, 4) if pos else 0.0,
                specificity=round((neg - fp) / neg, 4) if neg else 0.0,
                ppv=round(tp / (tp + fp), 4) if tp + fp else 0.0,
                brier_score=round(float(brier_score_loss(yt, pr)), 4),
            )
        )
    return result


def _gaps(metrics: list[GroupMetrics]) -> dict[str, float]:
    gaps = {}
    for field_name in ("sensitivity", "specificity", "ppv"):
        values = [getattr(m, field_name) for m in metrics]
        gaps[field_name] = round(max(values) - min(values), 4)
    return gaps


def evaluate(bundle: ModelBundle, X_test: pd.DataFrame, y_test: IntArray) -> dict[str, Any]:
    proba = bundle.pipeline.predict_proba(X_test[bundle.schema.all_features])[:, 1]
    threshold = bundle.decision_threshold
    age_band = pd.cut(X_test["age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)

    fairness: dict[str, Any] = {}
    for attribute, groups in (("gender", X_test["gender"]), ("age_band", age_band)):
        metrics = group_metrics(groups, y_test, proba, threshold)
        fairness[attribute] = {"groups": [asdict(m) for m in metrics], "max_gap": _gaps(metrics)}

    return {
        "model_version": bundle.model_version,
        "decision_threshold": round(threshold, 4),
        "n_test": int(len(y_test)),
        "calibration": asdict(calibration_report(y_test, proba)),
        "fairness": fairness,
    }


def _print_summary(report: dict[str, Any]) -> None:
    cal = report["calibration"]
    print(f"\nBrier {cal['brier_score']:.4f} (referência {cal['brier_reference']:.4f}) | "
          f"ECE {cal['expected_calibration_error']:.4f} | "
          f"média prevista {cal['mean_predicted']:.3f} x observada {cal['observed_rate']:.3f}")
    print("\n  prevista  observada     n")
    for b in cal["bins"]:
        print(f"  {b['mean_predicted']:>8.3f}  {b['observed_rate']:>9.3f}  {b['count']:>4}")
    for attribute, data in report["fairness"].items():
        print(f"\n{attribute}:  grupo      n   prev  AUC    sens   espec  VPP")
        for g in data["groups"]:
            auc = f"{g['auc_roc']:.3f}" if g["auc_roc"] is not None else "  -  "
            print(f"  {g['group']:>14} {g['n']:>5}  {g['prevalence']:.3f} {auc}  "
                  f"{g['sensitivity']:.3f}  {g['specificity']:.3f}  {g['ppv']:.3f}")
        print(f"  maior diferença: {data['max_gap']}")
    print()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Calibração e fairness no conjunto de teste.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--seed", type=int, default=42, help="a mesma usada no train_model")
    args = parser.parse_args()

    bundle = ModelLoader.load(args.models_dir / BUNDLE_FILENAME)
    df = CsvDatasetLoader.load(args.data)
    X = df[bundle.schema.all_features]
    y = df[bundle.schema.target].to_numpy(dtype=np.int64)
    _, X_test, _, y_test = split_holdout(X, y, TrainingConfig(seed=args.seed))

    report = evaluate(bundle, X_test, y_test)
    out = args.models_dir / EVALUATION_FILENAME
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    logger.info("Relatório salvo em %s", out)
    _print_summary(report)


if __name__ == "__main__":
    main()
