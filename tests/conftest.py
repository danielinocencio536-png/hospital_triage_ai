"""O modelo dos testes é treinado aqui, numa coorte pequena, para não depender do .joblib."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pytest

from src.pipelines.generate_mimic_data import MimicSyntheticDataGenerator, SyntheticDataConfig
from src.pipelines.train_model import (
    BUNDLE_FILENAME,
    ArtifactRepository,
    ModelBundle,
    ReadmissionModelTrainer,
    TrainingConfig,
)

TEST_SEED: Final[int] = 7

TEST_DATA_CONFIG: Final[SyntheticDataConfig] = SyntheticDataConfig(n_records=3_000, seed=TEST_SEED)

# versão reduzida para a suíte rodar em poucos segundos
FAST_TRAINING_CONFIG: Final[TrainingConfig] = TrainingConfig(
    cv_folds=3, n_estimators=80, shap_sample_size=200, seed=TEST_SEED
)

HIGH_RISK_PATIENT: Final[dict[str, Any]] = {
    "patient_id": "P-HIGH-001",
    "age": 84,
    "gender": "M",
    "admission_type": "EMERGENCY",
    "length_of_stay": 18.0,
    "num_lab_procedures": 95,
    "num_medications": 32,
    "comorbidities": {
        "hypertension": True,
        "diabetes": True,
        "heart_failure": True,
        "chronic_kidney_disease": True,
        "copd": True,
        "coronary_artery_disease": True,
        "cancer": False,
        "liver_disease": False,
    },
}

LOW_RISK_PATIENT: Final[dict[str, Any]] = {
    "patient_id": "P-LOW-001",
    "age": 24,
    "gender": "F",
    "admission_type": "ELECTIVE",
    "length_of_stay": 1.0,
    "num_lab_procedures": 10,
    "num_medications": 2,
    "comorbidities": {},
}


def make_patient(base: dict[str, Any] | None = None, **overrides: Any) -> dict[str, Any]:
    payload = copy.deepcopy(base if base is not None else HIGH_RISK_PATIENT)
    payload.update(overrides)
    return payload


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return MimicSyntheticDataGenerator(TEST_DATA_CONFIG).generate()


@pytest.fixture(scope="session")
def trained_artifacts(
    synthetic_df: pd.DataFrame,
) -> tuple[ModelBundle, dict[str, dict[str, float]]]:
    return ReadmissionModelTrainer(FAST_TRAINING_CONFIG).train(synthetic_df)


@pytest.fixture(scope="session")
def model_bundle(trained_artifacts: tuple[ModelBundle, dict[str, dict[str, float]]]) -> ModelBundle:
    return trained_artifacts[0]


@pytest.fixture(scope="session")
def models_dir(
    trained_artifacts: tuple[ModelBundle, dict[str, dict[str, float]]],
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    directory = tmp_path_factory.mktemp("models")
    ArtifactRepository(directory).save(*trained_artifacts)
    return directory


@pytest.fixture(scope="session")
def model_path(models_dir: Path) -> Path:
    return models_dir / BUNDLE_FILENAME
