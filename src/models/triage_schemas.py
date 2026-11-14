"""Contratos de entrada/saída da API de triagem (Pydantic v2).

Arquitetura BCE — camada Entity: estes modelos definem *o que* trafega pela
fronteira HTTP e são a ponte tipada entre o JSON do cliente e o ``DataFrame``
esperado pelo ``ModelBundle`` treinado no DIA 1.

Os nomes, domínios e faixas de valores espelham exatamente o gerador sintético
(``src/pipelines/generate_mimic_data.py``) e o ``FeatureSchema`` do treino:
    categóricas : gender, admission_type
    numéricas   : age, length_of_stay, num_lab_procedures, num_medications
    binárias    : comorbidity_<nome> (8 flags)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Final
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.pipelines.generate_mimic_data import COMORBIDITY_PREFIX

#: Identificador pseudonimizado — nunca nome, CPF ou prontuário em claro (LGPD).
PseudonymizedId = Annotated[
    str,
    Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$"),
]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]

_EXAMPLE_REQUEST: Final[dict[str, Any]] = {
    "patient_id": "P-10482931",
    "age": 78,
    "gender": "F",
    "admission_type": "EMERGENCY",
    "length_of_stay": 9.5,
    "num_lab_procedures": 62,
    "num_medications": 21,
    "comorbidities": {
        "hypertension": True,
        "diabetes": True,
        "heart_failure": True,
        "chronic_kidney_disease": False,
        "copd": False,
        "coronary_artery_disease": True,
        "cancer": False,
        "liver_disease": False,
    },
}


# =============================================================================
# ENUMS DE DOMÍNIO
# =============================================================================
class Gender(StrEnum):
    """Sexo biológico registrado (padrão MIMIC-IV)."""

    MALE = "M"
    FEMALE = "F"


class AdmissionType(StrEnum):
    """Tipo de admissão (padrão MIMIC-IV simplificado)."""

    EMERGENCY = "EMERGENCY"
    URGENT = "URGENT"
    ELECTIVE = "ELECTIVE"
    OBSERVATION = "OBSERVATION"


class RiskCategory(StrEnum):
    """Estratificação clínica do risco de readmissão em 30 dias."""

    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"

    @classmethod
    def from_probability(
        cls, probability: float, moderate_threshold: float, high_threshold: float
    ) -> RiskCategory:
        """Converte a probabilidade em categoria usando as faixas configuradas."""
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probabilidade fora de [0, 1]: {probability}")
        if probability >= high_threshold:
            return cls.HIGH
        if probability >= moderate_threshold:
            return cls.MODERATE
        return cls.LOW


class FactorDirection(StrEnum):
    """Sentido do efeito de um fator sobre o risco previsto."""

    INCREASES_RISK = "increases_risk"
    DECREASES_RISK = "decreases_risk"


# =============================================================================
# REQUEST
# =============================================================================
class _StrictModel(BaseModel):
    """Base: rejeita campos desconhecidos (evita typos silenciosos do cliente)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Comorbidities(_StrictModel):
    """Painel de comorbidades crônicas (ausência = ``False``)."""

    hypertension: bool = False
    diabetes: bool = False
    heart_failure: bool = False
    chronic_kidney_disease: bool = False
    copd: bool = Field(default=False, description="Doença pulmonar obstrutiva crônica.")
    coronary_artery_disease: bool = False
    cancer: bool = False
    liver_disease: bool = False

    @property
    def count(self) -> int:
        """Número total de comorbidades presentes."""
        return sum(self.model_dump().values())

    def to_feature_flags(self) -> dict[str, int]:
        """``{"comorbidity_diabetes": 1, ...}`` — formato das colunas de treino."""
        return {f"{COMORBIDITY_PREFIX}{name}": int(flag) for name, flag in self.model_dump().items()}


class PatientTriageRequest(_StrictModel):
    """Dados clínicos de uma internação no momento da alta."""

    model_config = ConfigDict(json_schema_extra={"examples": [_EXAMPLE_REQUEST]})

    patient_id: PseudonymizedId | None = Field(
        default=None, description="Identificador pseudonimizado, usado só para auditoria."
    )
    age: int = Field(ge=18, le=120, description="Idade em anos (coorte adulta, >= 18).")
    gender: Gender
    admission_type: AdmissionType
    length_of_stay: float = Field(
        gt=0.0, le=365.0, description="Tempo de internação em dias (aceita frações)."
    )
    num_lab_procedures: int = Field(ge=0, le=500, description="Exames laboratoriais na internação.")
    num_medications: int = Field(ge=0, le=200, description="Medicamentos distintos administrados.")
    comorbidities: Comorbidities = Field(default_factory=Comorbidities)

    def to_model_features(self) -> dict[str, Any]:
        """Achata o request no dicionário de features esperado pelo pipeline.

        As chaves batem 1:1 com ``FeatureSchema.all_features`` do DIA 1; a
        ordenação final das colunas é responsabilidade do serviço de inferência.
        """
        return {
            "gender": self.gender.value,
            "admission_type": self.admission_type.value,
            "age": self.age,
            "length_of_stay": self.length_of_stay,
            "num_lab_procedures": self.num_lab_procedures,
            "num_medications": self.num_medications,
            **self.comorbidities.to_feature_flags(),
        }


# =============================================================================
# RESPONSE
# =============================================================================
class RiskFactor(BaseModel):
    """Contribuição individual (SHAP local) de uma feature clínica."""

    model_config = ConfigDict(frozen=True)

    feature: str = Field(description="Nome da feature clínica original (one-hot agregado).")
    value: str | float | int | bool = Field(description="Valor informado para o paciente.")
    shap_value: float = Field(description="Contribuição em log-odds para o risco.")
    direction: FactorDirection

    @classmethod
    def from_shap(cls, feature: str, value: str | float | int | bool, shap_value: float) -> RiskFactor:
        direction = (
            FactorDirection.INCREASES_RISK if shap_value >= 0 else FactorDirection.DECREASES_RISK
        )
        return cls(feature=feature, value=value, shap_value=round(shap_value, 6), direction=direction)


class TriagePredictionResponse(BaseModel):
    """Resultado da triagem de risco de readmissão em 30 dias."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "request_id": "6f1c2b8e-3a4d-4f5e-9b7a-1c2d3e4f5a6b",
                    "patient_id": "P-10482931",
                    "risk_probability": 0.7124,
                    "risk_category": "High",
                    "readmission_predicted": True,
                    "decision_threshold": 0.5,
                    "top_risk_factors": [
                        {
                            "feature": "comorbidity_heart_failure",
                            "value": 1,
                            "shap_value": 0.41,
                            "direction": "increases_risk",
                        }
                    ],
                    "model_version": "1.0.0",
                    "predicted_at": "2026-11-12T14:03:22.518Z",
                }
            ]
        }
    )

    request_id: UUID = Field(default_factory=uuid4, description="Correlaciona com o log de auditoria.")
    patient_id: str | None = None
    risk_probability: Probability = Field(description="Probabilidade estimada de readmissão em 30 dias.")
    risk_category: RiskCategory
    readmission_predicted: bool = Field(
        description="True se risk_probability >= limiar de decisão do modelo (Youden)."
    )
    decision_threshold: Probability
    top_risk_factors: list[RiskFactor] = Field(
        default_factory=list, description="Fatores ordenados por |SHAP| decrescente."
    )
    model_version: str
    predicted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
