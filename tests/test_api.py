"""Testes da API com TestClient, usando SQLite no lugar do Postgres."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.v1.triage_router import AUDIT_HEADER, get_inference_service
from src.core.config import DatabaseSettings, Environment, Settings, get_settings
from src.database.connection import get_database_manager
from src.main import create_app
from src.models.triage_schemas import PatientTriageRequest, RiskCategory
from src.services.inference_service import InferenceError, InferenceResult
from tests.conftest import HIGH_RISK_PATIENT, LOW_RISK_PATIENT, make_patient

PREDICT_URL = "/api/v1/triage/predict"
HEALTH_URL = "/health"


def _clear_caches() -> None:
    get_settings.cache_clear()
    get_database_manager.cache_clear()


@pytest.fixture
def audit_db(tmp_path: Path) -> Path:
    return tmp_path / "triage_audit.db"


@pytest.fixture
def configure_env(
    monkeypatch: pytest.MonkeyPatch, audit_db: Path, model_path: Path
) -> Iterator[Callable[..., None]]:
    """Aponta para o modelo de teste e um SQLite temporário; devolve uma função para trocar mais variáveis."""

    def _configure(**overrides: str) -> None:
        env = {
            "TRIAGE_ENVIRONMENT": Environment.TESTING.value,
            "TRIAGE_LOG_LEVEL": "WARNING",
            "TRIAGE_MODEL_PATH": str(model_path),
            "TRIAGE_DATABASE__URL": f"sqlite+aiosqlite:///{audit_db.as_posix()}",
            "TRIAGE_CORS_ORIGINS": "http://localhost:3000",
            **overrides,
        }
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        _clear_caches()

    _configure()
    yield _configure
    _clear_caches()


@pytest.fixture
def make_client(configure_env: Callable[..., None]) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def _make(**env_overrides: str) -> TestClient:
        if env_overrides:
            configure_env(**env_overrides)
        client = TestClient(create_app())
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for client in reversed(clients):
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()


def _app(client: TestClient) -> FastAPI:
    app = client.app
    assert isinstance(app, FastAPI)
    return app


def _audit_rows(db_path: Path) -> list[tuple[Any, ...]]:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT request_id, patient_id, risk_category, risk_probability, "
            "model_version, input_features, top_risk_factors FROM triage_audit_logs"
        ).fetchall()


class TestHealth:
    def test_tudo_ok(self, client: TestClient) -> None:
        response = client.get(HEALTH_URL)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["environment"] == "testing"
        assert body["components"]["model_loaded"] is True
        assert body["components"]["model_version"] == "1.0.0"
        assert body["components"]["database"] == "ok"

    def test_banco_fora_deixa_api_degradada(self, client: TestClient) -> None:
        class _DownDatabase:
            async def ping(self) -> bool:
                return False

        real_database = _app(client).state.database
        _app(client).state.database = _DownDatabase()
        try:
            response = client.get(HEALTH_URL)
        finally:
            _app(client).state.database = real_database

        assert response.status_code == 200
        assert response.json()["status"] == "degraded"
        assert response.json()["components"]["database"] == "unavailable"

    def test_sem_modelo_retorna_503(self, make_client: Callable[..., TestClient], tmp_path: Path) -> None:
        client = make_client(TRIAGE_MODEL_PATH=str(tmp_path / "ausente.joblib"))
        response = client.get(HEALTH_URL)
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "unavailable"
        assert body["components"]["model_loaded"] is False
        assert body["components"]["model_version"] is None


class TestPredict:
    def test_paciente_alto_risco(self, client: TestClient) -> None:
        response = client.post(PREDICT_URL, json=HIGH_RISK_PATIENT)
        assert response.status_code == 200, response.text
        body = response.json()

        assert body["patient_id"] == HIGH_RISK_PATIENT["patient_id"]
        assert body["risk_category"] == RiskCategory.HIGH.value
        assert body["readmission_predicted"] is True
        assert body["risk_probability"] >= body["decision_threshold"]
        uuid.UUID(body["request_id"])

    def test_paciente_baixo_risco(self, client: TestClient) -> None:
        response = client.post(PREDICT_URL, json=LOW_RISK_PATIENT)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["risk_category"] == RiskCategory.LOW.value
        assert body["readmission_predicted"] is False

    def test_contrato_da_resposta(self, client: TestClient) -> None:
        body = client.post(PREDICT_URL, json=HIGH_RISK_PATIENT).json()
        assert set(body) == {
            "request_id",
            "patient_id",
            "risk_probability",
            "risk_category",
            "readmission_predicted",
            "decision_threshold",
            "top_risk_factors",
            "model_version",
            "predicted_at",
        }
        factors = body["top_risk_factors"]
        assert len(factors) == 5
        magnitudes = [abs(f["shap_value"]) for f in factors]
        assert magnitudes == sorted(magnitudes, reverse=True)
        for factor in factors:
            assert set(factor) == {"feature", "value", "shap_value", "direction"}
            assert factor["direction"] in {"increases_risk", "decreases_risk"}

    def test_categoria_coerente_com_probabilidade(self, client: TestClient) -> None:
        patient = make_patient(age=63, length_of_stay=5.0, admission_type="URGENT")
        body = client.post(PREDICT_URL, json=patient).json()
        expected = RiskCategory.from_probability(body["risk_probability"], 0.30, 0.50)
        assert body["risk_category"] == expected.value

    def test_patient_id_e_opcional(self, client: TestClient) -> None:
        patient = make_patient(LOW_RISK_PATIENT)
        del patient["patient_id"]
        response = client.post(PREDICT_URL, json=patient)
        assert response.status_code == 200
        assert response.json()["patient_id"] is None

    def test_grava_auditoria(self, client: TestClient, audit_db: Path) -> None:
        response = client.post(PREDICT_URL, json=HIGH_RISK_PATIENT)
        assert response.headers[AUDIT_HEADER] == "true"
        body = response.json()

        rows = _audit_rows(audit_db)
        assert len(rows) == 1
        request_id, patient_id, category, probability, version, features, factors = rows[0]
        assert uuid.UUID(request_id) == uuid.UUID(body["request_id"])
        assert patient_id == HIGH_RISK_PATIENT["patient_id"]
        assert category == body["risk_category"]
        assert probability == pytest.approx(body["risk_probability"])
        assert version == body["model_version"]
        assert '"comorbidity_heart_failure": 1' in features
        assert body["top_risk_factors"][0]["feature"] in factors

    def test_auditoria_fail_open(self, client: TestClient, audit_db: Path) -> None:
        class _FailingRepository:
            async def save(self, result: InferenceResult) -> None:
                raise ConnectionError("banco indisponível")

        _app(client).state.audit_repository = _FailingRepository()
        response = client.post(PREDICT_URL, json=HIGH_RISK_PATIENT)

        assert response.status_code == 200
        assert response.headers[AUDIT_HEADER] == "false"
        assert _audit_rows(audit_db) == []

    def test_sem_modelo_retorna_503(self, make_client: Callable[..., TestClient], tmp_path: Path) -> None:
        client = make_client(TRIAGE_MODEL_PATH=str(tmp_path / "ausente.joblib"))
        response = client.post(PREDICT_URL, json=HIGH_RISK_PATIENT)
        assert response.status_code == 503
        assert "indisponível" in response.json()["detail"]

    def test_falha_de_inferencia_retorna_500(self, client: TestClient, audit_db: Path) -> None:
        class _BrokenService:
            def predict(self, request: PatientTriageRequest) -> InferenceResult:
                raise InferenceError("Não foi possível calcular a predição.")

        _app(client).dependency_overrides[get_inference_service] = _BrokenService
        try:
            response = client.post(PREDICT_URL, json=HIGH_RISK_PATIENT)
        finally:
            _app(client).dependency_overrides.clear()

        assert response.status_code == 500
        assert response.json()["detail"] == "Não foi possível calcular a predição."
        assert _audit_rows(audit_db) == []


def _without(field: str) -> dict[str, Any]:
    payload = make_patient()
    del payload[field]
    return payload


class TestValidation:
    @pytest.mark.parametrize(
        ("payload", "bad_field"),
        [
            pytest.param(make_patient(age=17), "age", id="menor-de-idade"),
            pytest.param(make_patient(age=121), "age", id="idade-acima-do-limite"),
            pytest.param(make_patient(gender="X"), "gender", id="sexo-invalido"),
            pytest.param(make_patient(admission_type="TRAUMA"), "admission_type", id="admissao-invalida"),
            pytest.param(make_patient(length_of_stay=0), "length_of_stay", id="los-zero"),
            pytest.param(make_patient(num_medications=-1), "num_medications", id="medicacoes-negativas"),
            pytest.param(make_patient(patient_id="Maria da Silva"), "patient_id", id="id-nao-pseudonimizado"),
            pytest.param(make_patient(cpf="123.456.789-00"), "cpf", id="campo-extra"),
            pytest.param(
                make_patient(comorbidities={"asthma": True}), "asthma", id="comorbidade-desconhecida"
            ),
            pytest.param(_without("age"), "age", id="campo-obrigatorio-ausente"),
        ],
    )
    def test_payload_invalido(
        self, client: TestClient, audit_db: Path, payload: dict[str, Any], bad_field: str
    ) -> None:
        response = client.post(PREDICT_URL, json=payload)
        assert response.status_code == 422
        error_locations = [error["loc"] for error in response.json()["detail"]]
        assert any(bad_field in loc for loc in error_locations), error_locations
        assert _audit_rows(audit_db) == []

    def test_corpo_nao_json(self, client: TestClient) -> None:
        response = client.post(
            PREDICT_URL, content=b"age=80", headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 422


class TestHttpConfiguration:
    def test_cors_origem_permitida(self, client: TestClient) -> None:
        response = client.options(
            PREDICT_URL,
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

    def test_cors_origem_bloqueada(self, client: TestClient) -> None:
        response = client.options(
            PREDICT_URL,
            headers={"Origin": "https://site-malicioso.example", "Access-Control-Request-Method": "POST"},
        )
        assert "access-control-allow-origin" not in response.headers

    def test_cors_expoe_header_de_auditoria(self, client: TestClient) -> None:
        response = client.post(
            PREDICT_URL, json=LOW_RISK_PATIENT, headers={"Origin": "http://localhost:3000"}
        )
        assert AUDIT_HEADER.lower() in response.headers["access-control-expose-headers"].lower()

    def test_docs_disponiveis_fora_de_producao(self, client: TestClient) -> None:
        assert client.get("/docs").status_code == 200
        assert PREDICT_URL in client.get("/openapi.json").json()["paths"]

    def test_docs_desabilitadas_em_producao(self, configure_env: Callable[..., None]) -> None:
        settings = Settings(
            environment=Environment.PRODUCTION,
            database=DatabaseSettings(password="senha-forte-de-producao"),
        )
        # sem o with o lifespan não roda, e aqui só interessam as rotas
        client = TestClient(create_app(settings))
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
