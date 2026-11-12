"""Gerador de dados sintéticos inspirados na estrutura do MIMIC-IV.

Arquitetura BCE (Boundary-Control-Entity):
    * Entity   -> ``ComorbiditySpec`` e ``SyntheticDataConfig``: estruturas imutáveis
                  que descrevem o domínio clínico e os parâmetros da simulação.
    * Control  -> ``MimicSyntheticDataGenerator``: regra de negócio que gera a
                  coorte com correlações estatísticas clinicamente plausíveis.
    * Boundary -> ``CsvDatasetWriter`` + ``main()``: fronteira de I/O (disco/CLI).

Correlações modeladas (direção clínica esperada):
    * Idade ↑  -> prevalência de comorbidades ↑
    * Idade ↑ e nº de comorbidades ↑ -> tempo de internação (LOS) ↑
    * LOS ↑ -> nº de procedimentos laboratoriais ↑
    * Comorbidades ↑ -> nº de medicações (polifarmácia) ↑
    * Todos os fatores acima (+ admissão de emergência) -> readmissão em 30 dias ↑

A prevalência do alvo ``readmitted_30d`` é calibrada por bisseção no intercepto
do modelo logístico gerador, garantindo a taxa desejada (padrão 15%, dentro da
faixa de 12-18% observada na literatura).

Execução:
    python -m src.pipelines.generate_mimic_data --n-records 10000 --seed 42
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: Prefixo das colunas binárias que compõem o "array" de comorbidades.
COMORBIDITY_PREFIX: Final[str] = "comorbidity_"
TARGET_COLUMN: Final[str] = "readmitted_30d"
DEFAULT_OUTPUT_PATH: Final[Path] = Path("data/raw/mimic_synthetic.csv")


# =============================================================================
# ENTITY
# =============================================================================
@dataclass(frozen=True, slots=True)
class ComorbiditySpec:
    """Descreve uma comorbidade e seu comportamento epidemiológico simulado.

    Attributes:
        name: Identificador snake_case (vira coluna ``comorbidity_<name>``).
        base_logit: Log-odds de prevalência para um paciente de 18 anos.
        age_slope: Incremento de log-odds por década acima de 18 anos.
        readmission_weight: Contribuição (log-odds) para o risco de readmissão.
    """

    name: str
    base_logit: float
    age_slope: float
    readmission_weight: float


#: Painel de comorbidades crônicas frequentes em coortes de readmissão
#: (inspirado nos componentes do Índice de Charlson / escore LACE).
DEFAULT_COMORBIDITIES: Final[tuple[ComorbiditySpec, ...]] = (
    ComorbiditySpec("hypertension", base_logit=-2.6, age_slope=0.55, readmission_weight=0.15),
    ComorbiditySpec("diabetes", base_logit=-3.0, age_slope=0.40, readmission_weight=0.30),
    ComorbiditySpec("heart_failure", base_logit=-4.8, age_slope=0.55, readmission_weight=0.70),
    ComorbiditySpec("chronic_kidney_disease", base_logit=-4.5, age_slope=0.50, readmission_weight=0.55),
    ComorbiditySpec("copd", base_logit=-4.2, age_slope=0.40, readmission_weight=0.50),
    ComorbiditySpec("coronary_artery_disease", base_logit=-4.3, age_slope=0.50, readmission_weight=0.35),
    ComorbiditySpec("cancer", base_logit=-4.0, age_slope=0.35, readmission_weight=0.60),
    ComorbiditySpec("liver_disease", base_logit=-4.0, age_slope=0.10, readmission_weight=0.45),
)

#: Tipos de admissão no padrão MIMIC-IV (simplificado) e suas probabilidades.
ADMISSION_TYPES: Final[dict[str, float]] = {
    "EMERGENCY": 0.60,
    "URGENT": 0.15,
    "ELECTIVE": 0.20,
    "OBSERVATION": 0.05,
}

#: Contribuição do tipo de admissão para o risco de readmissão (log-odds).
ADMISSION_RISK: Final[dict[str, float]] = {
    "EMERGENCY": 0.35,
    "URGENT": 0.20,
    "ELECTIVE": -0.40,
    "OBSERVATION": -0.20,
}


@dataclass(frozen=True, slots=True)
class SyntheticDataConfig:
    """Parâmetros de simulação da coorte.

    Attributes:
        n_records: Quantidade de pacientes/internações a gerar.
        seed: Semente para reprodutibilidade total.
        target_prevalence: Taxa alvo de readmissão em 30 dias (0.12-0.18).
        min_age / max_age: Faixa etária adulta (MIMIC-IV inclui apenas >= 18).
        comorbidities: Painel de comorbidades simuladas.
    """

    n_records: int = 10_000
    seed: int = 42
    target_prevalence: float = 0.15
    min_age: int = 18
    max_age: int = 95
    comorbidities: tuple[ComorbiditySpec, ...] = field(default=DEFAULT_COMORBIDITIES)

    def __post_init__(self) -> None:
        if self.n_records <= 0:
            raise ValueError("n_records deve ser positivo.")
        if not 0.12 <= self.target_prevalence <= 0.18:
            raise ValueError("target_prevalence deve estar na faixa clínica de 0.12 a 0.18.")
        if not 18 <= self.min_age < self.max_age <= 120:
            raise ValueError("Faixa etária inválida (esperado 18 <= min_age < max_age <= 120).")


# =============================================================================
# CONTROL
# =============================================================================
class MimicSyntheticDataGenerator:
    """Gera uma coorte sintética com correlações clínicas coerentes.

    Cada etapa da simulação é um método privado com responsabilidade única
    (SRP), e o gerador depende apenas da configuração injetada (DIP).
    """

    def __init__(self, config: SyntheticDataConfig) -> None:
        self._config = config
        self._rng: np.random.Generator = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------ API
    def generate(self) -> pd.DataFrame:
        """Executa a simulação completa e retorna o DataFrame da coorte."""
        n = self._config.n_records
        logger.info("Gerando %d registros sintéticos (seed=%d)...", n, self._config.seed)

        age = self._sample_age(n)
        gender = self._rng.choice(np.array(["M", "F"]), size=n, p=[0.52, 0.48])
        admission_type = self._rng.choice(
            np.array(list(ADMISSION_TYPES.keys())), size=n, p=list(ADMISSION_TYPES.values())
        )
        comorbidity_matrix = self._sample_comorbidities(age)
        n_comorbidities = comorbidity_matrix.sum(axis=1)

        length_of_stay = self._sample_length_of_stay(age, n_comorbidities, admission_type)
        num_lab_procedures = self._sample_lab_procedures(length_of_stay)
        num_medications = self._sample_medications(n_comorbidities, age)

        readmission_logit = self._readmission_linear_predictor(
            age=age,
            gender=gender,
            admission_type=admission_type,
            comorbidity_matrix=comorbidity_matrix,
            length_of_stay=length_of_stay,
            num_lab_procedures=num_lab_procedures,
            num_medications=num_medications,
        )
        readmitted = self._sample_target(readmission_logit)

        df = pd.DataFrame(
            {
                "patient_id": self._build_patient_ids(n),
                "age": age,
                "gender": gender,
                "admission_type": admission_type,
                "length_of_stay": length_of_stay,
                "num_lab_procedures": num_lab_procedures,
                "num_medications": num_medications,
            }
        )
        for idx, spec in enumerate(self._config.comorbidities):
            df[f"{COMORBIDITY_PREFIX}{spec.name}"] = comorbidity_matrix[:, idx].astype(np.int8)
        df[TARGET_COLUMN] = readmitted.astype(np.int8)

        logger.info("Prevalência de readmissão obtida: %.2f%%", 100 * df[TARGET_COLUMN].mean())
        return df

    # ------------------------------------------------------------- Etapas
    def _sample_age(self, n: int) -> IntArray:
        """Idade com distribuição assimétrica à esquerda (coorte hospitalar envelhecida)."""
        cfg = self._config
        span = cfg.max_age - cfg.min_age
        # Beta(3.2, 2.0) concentra a massa entre ~55 e ~80 anos.
        raw = self._rng.beta(3.2, 2.0, size=n)
        return (cfg.min_age + raw * span).round().astype(np.int64)

    def _sample_comorbidities(self, age: IntArray) -> IntArray:
        """Matriz binária (n x k): prevalência cresce com a idade (logística)."""
        decades_above_min = (age - self._config.min_age) / 10.0
        columns: list[IntArray] = []
        for spec in self._config.comorbidities:
            logit = spec.base_logit + spec.age_slope * decades_above_min
            prob = _sigmoid(logit)
            columns.append(self._rng.binomial(1, prob).astype(np.int64))
        return np.column_stack(columns)

    def _sample_length_of_stay(
        self,
        age: IntArray,
        n_comorbidities: IntArray,
        admission_type: npt.NDArray[np.str_],
    ) -> FloatArray:
        """LOS (dias) ~ LogNormal: média cresce com idade, carga de doença e urgência."""
        emergency_boost = np.where(np.isin(admission_type, ["EMERGENCY", "URGENT"]), 0.20, 0.0)
        mu = 0.85 + 0.008 * (age - self._config.min_age) + 0.16 * n_comorbidities + emergency_boost
        los = self._rng.lognormal(mean=mu, sigma=0.55)
        # Limites plausíveis: 0.5 dia (observação curta) até 60 dias.
        return np.clip(los, 0.5, 60.0).round(1)

    def _sample_lab_procedures(self, length_of_stay: FloatArray) -> IntArray:
        """Exames laboratoriais ~ Poisson, proporcionais (sublinearmente) ao LOS."""
        lam = 8.0 + 9.0 * np.sqrt(length_of_stay)
        return np.clip(self._rng.poisson(lam), 1, 200).astype(np.int64)

    def _sample_medications(self, n_comorbidities: IntArray, age: IntArray) -> IntArray:
        """Medicações ~ Poisson: polifarmácia associada a comorbidades e idade."""
        lam = 4.0 + 2.6 * n_comorbidities + 0.05 * (age - self._config.min_age)
        return np.clip(self._rng.poisson(lam), 0, 80).astype(np.int64)

    def _readmission_linear_predictor(
        self,
        *,
        age: IntArray,
        gender: npt.NDArray[np.str_],
        admission_type: npt.NDArray[np.str_],
        comorbidity_matrix: IntArray,
        length_of_stay: FloatArray,
        num_lab_procedures: IntArray,
        num_medications: IntArray,
    ) -> FloatArray:
        """Log-odds de readmissão (sem intercepto) — o "ground truth" da simulação.

        Inclui um termo de interação idade x comorbidades e um efeito não
        linear do LOS (log), para que modelos baseados em árvore tenham sinal
        além do puramente linear.
        """
        weights = np.array([s.readmission_weight for s in self._config.comorbidities])
        n_comorb = comorbidity_matrix.sum(axis=1)
        elderly = (age >= 65).astype(np.float64)

        logit = (
            0.018 * (age - 60)
            + comorbidity_matrix @ weights
            + 0.25 * elderly * np.minimum(n_comorb, 4)  # interação idoso x multimorbidade
            + 0.45 * np.log1p(length_of_stay)
            + 0.004 * (num_lab_procedures - 30)
            + 0.025 * (num_medications - 10)
            + np.where(gender == "M", 0.08, 0.0)
            + np.vectorize(ADMISSION_RISK.__getitem__)(admission_type)
        )
        # Ruído não observado (fatores sociais, adesão, etc.)
        return logit + self._rng.normal(0.0, 0.35, size=logit.shape[0])

    def _sample_target(self, linear_predictor: FloatArray) -> IntArray:
        """Calibra o intercepto por bisseção para atingir a prevalência alvo."""
        target = self._config.target_prevalence
        low, high = -15.0, 15.0
        for _ in range(60):
            mid = (low + high) / 2.0
            if _sigmoid(linear_predictor + mid).mean() > target:
                high = mid
            else:
                low = mid
        intercept = (low + high) / 2.0
        probs = _sigmoid(linear_predictor + intercept)
        logger.debug("Intercepto calibrado: %.4f", intercept)
        return self._rng.binomial(1, probs).astype(np.int64)

    def _build_patient_ids(self, n: int) -> npt.NDArray[np.str_]:
        """IDs no estilo MIMIC (subject_id de 8 dígitos), únicos."""
        ids = self._rng.choice(np.arange(10_000_000, 20_000_000), size=n, replace=False)
        return ids.astype(str)


def _sigmoid(x: FloatArray) -> FloatArray:
    """Função logística numericamente estável."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# =============================================================================
# BOUNDARY
# =============================================================================
class DatasetWriter(Protocol):
    """Contrato de persistência (permite trocar CSV por Parquet/DB — OCP/DIP)."""

    def write(self, df: pd.DataFrame, path: Path) -> Path: ...


class CsvDatasetWriter:
    """Persiste o dataset em CSV, criando diretórios quando necessário."""

    def write(self, df: pd.DataFrame, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Dataset salvo em %s (%d linhas, %d colunas)", path, *df.shape)
        return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera dados sintéticos estilo MIMIC-IV.")
    parser.add_argument("--n-records", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prevalence", type=float, default=0.15)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Ponto de entrada CLI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()
    config = SyntheticDataConfig(
        n_records=args.n_records, seed=args.seed, target_prevalence=args.prevalence
    )
    df = MimicSyntheticDataGenerator(config).generate()
    CsvDatasetWriter().write(df, args.output)


if __name__ == "__main__":
    main()
