"""Testes do gerador de dados, do treino e do serviço de inferência (sem HTTP)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from pydantic import ValidationError

from src.core.config import RiskStratificationSettings
from src.pipelines.generate_mimic_data import (
    ADMISSION_TYPES,
    COMORBIDITY_PREFIX,
    DEFAULT_COMORBIDITIES,
    TARGET_COLUMN,
    CsvDatasetWriter,
    MimicSyntheticDataGenerator,
    SyntheticDataConfig,
)
from src.pipelines.train_model import (
    METRICS_FILENAME,
    MODEL_VERSION,
    SHAP_FILENAME,
    ClinicalEvaluator,
    CsvDatasetLoader,
    FeatureSchema,
    ModelBundle,
    ThresholdSelector,
    _build_feature_name_map,
)
from src.models.triage_schemas import (
    FactorDirection,
    PatientTriageRequest,
    RiskCategory,
)
from src.services.inference_service import (
    ModelLoader,
    ModelLoadError,
    TriageInferenceService,
)
from tests.conftest import (
    HIGH_RISK_PATIENT,
    LOW_RISK_PATIENT,
    TEST_DATA_CONFIG,
    TEST_SEED,
    make_patient,
)

COMORBIDITY_COLUMNS = [f"{COMORBIDITY_PREFIX}{spec.name}" for spec in DEFAULT_COMORBIDITIES]
EXPECTED_COLUMNS = {
    "patient_id",
    "age",
    "gender",
    "admission_type",
    "length_of_stay",
    "num_lab_procedures",
    "num_medications",
    *COMORBIDITY_COLUMNS,
    TARGET_COLUMN,
}


class TestSyntheticDataConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"n_records": 0}, id="n_records-zero"),
            pytest.param({"target_prevalence": 0.05}, id="prevalencia-baixa"),
            pytest.param({"target_prevalence": 0.30}, id="prevalencia-alta"),
            pytest.param({"min_age": 10}, id="menor-de-idade"),
            pytest.param({"min_age": 60, "max_age": 50}, id="faixa-invertida"),
        ],
    )
    def test_rejeita_parametros_invalidos(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            SyntheticDataConfig(**kwargs)  # type: ignore[arg-type]


class TestMimicSyntheticDataGenerator:
    def test_contrato_do_dataset(self, synthetic_df: pd.DataFrame) -> None:
        assert set(synthetic_df.columns) == EXPECTED_COLUMNS
        assert len(synthetic_df) == TEST_DATA_CONFIG.n_records
        assert not synthetic_df.isna().any().any()
        assert synthetic_df["patient_id"].is_unique

    def test_dominios_e_faixas(self, synthetic_df: pd.DataFrame) -> None:
        assert synthetic_df["age"].between(TEST_DATA_CONFIG.min_age, TEST_DATA_CONFIG.max_age).all()
        assert set(synthetic_df["gender"]) <= {"M", "F"}
        assert set(synthetic_df["admission_type"]) <= set(ADMISSION_TYPES)
        assert synthetic_df["length_of_stay"].between(0.5, 60.0).all()
        assert (synthetic_df["num_lab_procedures"] >= 1).all()
        assert (synthetic_df["num_medications"] >= 0).all()
        for column in [*COMORBIDITY_COLUMNS, TARGET_COLUMN]:
            assert set(synthetic_df[column].unique()) <= {0, 1}, column

    def test_prevalencia_proxima_do_alvo(self, synthetic_df: pd.DataFrame) -> None:
        prevalence = synthetic_df[TARGET_COLUMN].mean()
        assert prevalence == pytest.approx(TEST_DATA_CONFIG.target_prevalence, abs=0.03)

    def test_reprodutivel_com_mesma_semente(self) -> None:
        config = SyntheticDataConfig(n_records=300, seed=TEST_SEED)
        first = MimicSyntheticDataGenerator(config).generate()
        second = MimicSyntheticDataGenerator(config).generate()
        pd.testing.assert_frame_equal(first, second)

    def test_sementes_diferentes_geram_coortes_diferentes(self) -> None:
        first = MimicSyntheticDataGenerator(SyntheticDataConfig(n_records=300, seed=1)).generate()
        second = MimicSyntheticDataGenerator(SyntheticDataConfig(n_records=300, seed=2)).generate()
        assert not first.equals(second)

    def test_correlacoes_clinicas(self, synthetic_df: pd.DataFrame) -> None:
        n_comorbidities = synthetic_df[COMORBIDITY_COLUMNS].sum(axis=1)
        assert synthetic_df["age"].corr(n_comorbidities) > 0.2
        assert synthetic_df["length_of_stay"].corr(synthetic_df["num_lab_procedures"]) > 0.3
        assert n_comorbidities.corr(synthetic_df["num_medications"]) > 0.3

    def test_insuficiencia_cardiaca_aumenta_readmissao(self, synthetic_df: pd.DataFrame) -> None:
        rate = synthetic_df.groupby("comorbidity_heart_failure")[TARGET_COLUMN].mean()
        assert rate[1] > rate[0]

    def test_csv_writer_e_loader_ida_e_volta(self, synthetic_df: pd.DataFrame, tmp_path: Path) -> None:
        path = CsvDatasetWriter().write(synthetic_df.head(50), tmp_path / "raw" / "cohort.csv")
        loaded = CsvDatasetLoader.load(path)
        assert loaded.shape == (50, synthetic_df.shape[1])
        # patient_id tem que voltar como string
        assert loaded["patient_id"].map(type).eq(str).all()

    def test_loader_falha_sem_dataset(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="generate_mimic_data"):
            CsvDatasetLoader.load(tmp_path / "inexistente.csv")


class TestFeatureSchema:
    def test_deriva_esquema_do_dataframe(self, synthetic_df: pd.DataFrame) -> None:
        schema = FeatureSchema.from_dataframe(synthetic_df)
        assert schema.categorical == ("gender", "admission_type")
        assert schema.binary == tuple(sorted(COMORBIDITY_COLUMNS))
        assert schema.all_features == [*schema.categorical, *schema.numerical, *schema.binary]
        assert TARGET_COLUMN not in schema.all_features
        assert "patient_id" not in schema.all_features

    def test_falha_sem_colunas_obrigatorias(self, synthetic_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="age"):
            FeatureSchema.from_dataframe(synthetic_df.drop(columns=["age"]))

    def test_falha_sem_comorbidades(self, synthetic_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match=COMORBIDITY_PREFIX):
            FeatureSchema.from_dataframe(synthetic_df.drop(columns=COMORBIDITY_COLUMNS))


class TestThresholdAndMetrics:
    def test_youden_separa_classes_perfeitamente_separaveis(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        y_proba = np.array([0.05, 0.10, 0.20, 0.70, 0.80, 0.95])
        threshold = ThresholdSelector.youden(y_true, y_proba)
        assert 0.0 <= threshold <= 1.0
        assert ((y_proba >= threshold).astype(int) == y_true).all()

    def test_metricas_clinicas_de_matriz_conhecida(self) -> None:
        # Limiar 0.5 -> TP=2, FN=1, TN=3, FP=1
        y_true = np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.int64)
        y_proba = np.array([0.9, 0.8, 0.3, 0.6, 0.2, 0.1, 0.4])
        metrics = ClinicalEvaluator.evaluate(y_true, y_proba, threshold=0.5)
        assert (metrics.true_positives, metrics.false_negatives) == (2, 1)
        assert (metrics.true_negatives, metrics.false_positives) == (3, 1)
        assert metrics.sensitivity_recall == pytest.approx(2 / 3, abs=1e-4)
        assert metrics.specificity == pytest.approx(3 / 4, abs=1e-4)
        assert metrics.ppv_precision == pytest.approx(2 / 3, abs=1e-4)
        assert metrics.npv == pytest.approx(3 / 4, abs=1e-4)
        assert metrics.prevalence == pytest.approx(3 / 7, abs=1e-4)

    def test_mapa_de_features_agrega_one_hot(self, synthetic_df: pd.DataFrame) -> None:
        schema = FeatureSchema.from_dataframe(synthetic_df)
        mapping = _build_feature_name_map(
            ["gender_F", "gender_M", "admission_type_EMERGENCY", "age", "comorbidity_copd"], schema
        )
        assert mapping == {
            "gender_F": "gender",
            "gender_M": "gender",
            "admission_type_EMERGENCY": "admission_type",
            "age": "age",
            "comorbidity_copd": "comorbidity_copd",
        }


class TestReadmissionModelTrainer:
    def test_bundle_completo(self, model_bundle: ModelBundle) -> None:
        assert 0.0 < model_bundle.decision_threshold < 1.0
        assert model_bundle.model_version == MODEL_VERSION
        assert set(model_bundle.pipeline.named_steps) == {"preprocessor", "classifier"}
        assert set(model_bundle.feature_name_map) == set(model_bundle.transformed_feature_names)
        assert set(model_bundle.feature_name_map.values()) == set(model_bundle.schema.all_features)
        assert {"scikit-learn", "xgboost", "shap"} <= set(model_bundle.library_versions)

    def test_modelo_aprende_sinal_clinico(self, model_bundle: ModelBundle) -> None:
        metrics = model_bundle.metrics
        assert metrics.auc_roc > 0.65, "AUC muito baixa: o modelo não aprendeu o sinal."
        assert metrics.auc_pr > metrics.prevalence, "AUC-PR deveria superar o classificador aleatório."
        assert 0.0 < metrics.brier_score < 0.25

    def test_importancia_shap_global(
        self, trained_artifacts: tuple[ModelBundle, dict[str, dict[str, float]]]
    ) -> None:
        bundle, importance = trained_artifacts
        assert set(importance) == {"transformed", "original"}
        assert set(importance["original"]) == set(bundle.schema.all_features)
        values = list(importance["original"].values())
        assert values == sorted(values, reverse=True)
        assert all(v >= 0 for v in values)


class TestArtifacts:
    def test_artefatos_salvos(self, models_dir: Path, model_bundle: ModelBundle) -> None:
        report = json.loads((models_dir / METRICS_FILENAME).read_text("utf-8"))
        assert report["model_version"] == model_bundle.model_version
        assert report["decision_threshold"] == pytest.approx(model_bundle.decision_threshold)
        assert report["holdout_metrics"]["auc_roc"] == model_bundle.metrics.auc_roc

        shap_report = json.loads((models_dir / SHAP_FILENAME).read_text("utf-8"))
        assert set(shap_report["original"]) == set(model_bundle.schema.all_features)

    def test_model_loader_ida_e_volta(self, model_path: Path, model_bundle: ModelBundle) -> None:
        loaded = ModelLoader.load(model_path)
        assert isinstance(loaded, ModelBundle)
        assert loaded.decision_threshold == pytest.approx(model_bundle.decision_threshold)
        assert loaded.schema == model_bundle.schema

    def test_model_loader_arquivo_inexistente(self, tmp_path: Path) -> None:
        with pytest.raises(ModelLoadError, match="não encontrado"):
            ModelLoader.load(tmp_path / "nao_existe.joblib")

    def test_model_loader_tipo_invalido(self, tmp_path: Path) -> None:
        path = tmp_path / "errado.joblib"
        joblib.dump({"not": "a bundle"}, path)
        with pytest.raises(ModelLoadError, match="esperado ModelBundle"):
            ModelLoader.load(path)

    def test_model_loader_arquivo_corrompido(self, tmp_path: Path) -> None:
        path = tmp_path / "corrompido.joblib"
        path.write_bytes(b"isto nao e um pickle")
        with pytest.raises(ModelLoadError, match="desserializar"):
            ModelLoader.load(path)


@pytest.fixture(scope="module")
def risk_settings() -> RiskStratificationSettings:
    return RiskStratificationSettings()


@pytest.fixture(scope="module")
def service(model_bundle: ModelBundle, risk_settings: RiskStratificationSettings) -> TriageInferenceService:
    return TriageInferenceService(model_bundle, risk_settings)


class TestTriageInferenceService:
    def test_predicao_consistente(
        self, service: TriageInferenceService, risk_settings: RiskStratificationSettings
    ) -> None:
        result = service.predict(PatientTriageRequest.model_validate(HIGH_RISK_PATIENT))
        response = result.response

        assert 0.0 <= response.risk_probability <= 1.0
        assert response.risk_category == RiskCategory.from_probability(
            response.risk_probability,
            risk_settings.moderate_threshold,
            risk_settings.high_threshold,
        )
        assert response.readmission_predicted == (
            response.risk_probability >= response.decision_threshold
        )
        assert response.patient_id == HIGH_RISK_PATIENT["patient_id"]
        assert response.model_version == MODEL_VERSION
        assert result.latency_ms > 0
        assert result.features["comorbidity_heart_failure"] == 1

    def test_estratifica_alto_e_baixo_risco(self, service: TriageInferenceService) -> None:
        high = service.predict(PatientTriageRequest.model_validate(HIGH_RISK_PATIENT)).response
        low = service.predict(PatientTriageRequest.model_validate(LOW_RISK_PATIENT)).response
        assert high.risk_probability > low.risk_probability
        assert high.risk_category is RiskCategory.HIGH
        assert low.risk_category is RiskCategory.LOW

    def test_coerente_com_o_pipeline_treinado(
        self, service: TriageInferenceService, model_bundle: ModelBundle
    ) -> None:
        request = PatientTriageRequest.model_validate(HIGH_RISK_PATIENT)
        features = request.to_model_features()
        frame = pd.DataFrame([features])[model_bundle.schema.all_features]
        expected = float(model_bundle.pipeline.predict_proba(frame)[0, 1])
        assert service.predict(request).response.risk_probability == pytest.approx(expected, abs=1e-6)

    def test_deterministica(self, service: TriageInferenceService) -> None:
        request = PatientTriageRequest.model_validate(LOW_RISK_PATIENT)
        first = service.predict(request).response
        second = service.predict(request).response
        assert first.risk_probability == second.risk_probability
        assert first.top_risk_factors == second.top_risk_factors
        assert first.request_id != second.request_id

    def test_top_fatores_ordenados_por_impacto(
        self, service: TriageInferenceService, risk_settings: RiskStratificationSettings
    ) -> None:
        request = PatientTriageRequest.model_validate(HIGH_RISK_PATIENT)
        factors = service.predict(request).response.top_risk_factors
        features = request.to_model_features()

        assert len(factors) == risk_settings.top_k_factors
        magnitudes = [abs(f.shap_value) for f in factors]
        assert magnitudes == sorted(magnitudes, reverse=True)
        for factor in factors:
            assert factor.feature in features
            assert factor.value == features[factor.feature]
            expected = (
                FactorDirection.INCREASES_RISK if factor.shap_value >= 0 else FactorDirection.DECREASES_RISK
            )
            assert factor.direction is expected

    def test_top_k_configuravel(self, model_bundle: ModelBundle) -> None:
        service = TriageInferenceService(model_bundle, RiskStratificationSettings(top_k_factors=2))
        response = service.predict(PatientTriageRequest.model_validate(HIGH_RISK_PATIENT)).response
        assert len(response.top_risk_factors) == 2

    def test_shap_local_e_aditivo(self, model_bundle: ModelBundle) -> None:
        # SHAP é aditivo: base + soma dos valores = logit(p)
        all_features = len(model_bundle.schema.all_features)
        service = TriageInferenceService(
            model_bundle, RiskStratificationSettings(top_k_factors=all_features)
        )
        patient = make_patient(age=67, length_of_stay=6.0, admission_type="URGENT")
        response = service.predict(PatientTriageRequest.model_validate(patient)).response

        assert len(response.top_risk_factors) == all_features

        # A base vem do próprio XGBoost (pred_contribs). No shap 0.54 o expected_value
        # do TreeExplainer só é preenchido depois da primeira chamada a shap_values.
        preprocessor = model_bundle.pipeline.named_steps["preprocessor"]
        classifier = model_bundle.pipeline.named_steps["classifier"]
        patient_features = PatientTriageRequest.model_validate(patient).to_model_features()
        frame = pd.DataFrame([patient_features])
        transformed = preprocessor.transform(frame[model_bundle.schema.all_features])
        contribs = classifier.get_booster().predict(xgb.DMatrix(transformed), pred_contribs=True)
        base_value = float(contribs[0, -1])

        shap_sum = sum(f.shap_value for f in response.top_risk_factors)
        assert {f.feature for f in response.top_risk_factors} == set(patient_features)
        p = response.risk_probability
        assert base_value + shap_sum == pytest.approx(math.log(p / (1 - p)), abs=1e-3)


class TestRiskStratification:
    @pytest.mark.parametrize(
        ("probability", "expected"),
        [
            (0.0, RiskCategory.LOW),
            (0.2999, RiskCategory.LOW),
            (0.30, RiskCategory.MODERATE),
            (0.4999, RiskCategory.MODERATE),
            (0.50, RiskCategory.HIGH),
            (1.0, RiskCategory.HIGH),
        ],
    )
    def test_faixas_de_risco(self, probability: float, expected: RiskCategory) -> None:
        assert RiskCategory.from_probability(probability, 0.30, 0.50) is expected

    @pytest.mark.parametrize("probability", [-0.01, 1.01])
    def test_probabilidade_fora_do_intervalo(self, probability: float) -> None:
        with pytest.raises(ValueError):
            RiskCategory.from_probability(probability, 0.30, 0.50)

    def test_limiares_fora_de_ordem(self) -> None:
        with pytest.raises(ValidationError, match="moderate_threshold"):
            RiskStratificationSettings(moderate_threshold=0.6, high_threshold=0.5)

    def test_request_gera_todas_as_features_do_modelo(self, model_bundle: ModelBundle) -> None:
        features = PatientTriageRequest.model_validate(LOW_RISK_PATIENT).to_model_features()
        assert set(features) == set(model_bundle.schema.all_features)
        # comorbidade omitida vira 0
        assert all(features[column] == 0 for column in COMORBIDITY_COLUMNS)
