"""Configuração centralizada da aplicação (12-factor: tudo via ambiente/.env).

Arquitetura BCE:
    * Entity -> ``Settings`` e seus sub-blocos (``DatabaseSettings``,
      ``RiskStratificationSettings``): contrato tipado e validado de configuração.
    * Boundary -> ``get_settings()``: único ponto de acesso, com cache, usado
      pelos demais módulos (e sobrescrevível em testes via ``cache_clear()``).

Convenção de variáveis de ambiente (prefixo ``TRIAGE_``, aninhamento por ``__``):
    TRIAGE_ENVIRONMENT=production
    TRIAGE_DATABASE__HOST=postgres
    TRIAGE_DATABASE__PASSWORD=s3cr3t
    TRIAGE_RISK__MODERATE_THRESHOLD=0.30

Para testes locais sem PostgreSQL basta definir uma URL completa:
    TRIAGE_DATABASE__URL=sqlite+aiosqlite:///./triage_dev.db
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Final, Self
from urllib.parse import quote_plus

from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveInt,
    SecretStr,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

#: Raiz do projeto (…/hospital_triage_ai) — resolve caminhos relativos de forma
#: independente do diretório de onde o processo foi iniciado.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

Probability = Annotated[float, Field(gt=0.0, lt=1.0)]


# =============================================================================
# ENTITY
# =============================================================================
class Environment(StrEnum):
    """Ambientes de execução suportados."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class DatabaseSettings(BaseModel):
    """Parâmetros de conexão com o PostgreSQL (driver assíncrono ``asyncpg``).

    Se ``url`` for informada, ela tem precedência sobre os campos individuais —
    útil para SQLite em testes ou para DSNs gerenciados (ex.: RDS, Cloud SQL).
    """

    url: SecretStr | None = None
    driver: str = "postgresql+asyncpg"
    host: str = "localhost"
    port: Annotated[int, Field(ge=1, le=65_535)] = 5432
    user: str = "triage"
    password: SecretStr = SecretStr("triage")
    name: str = "triage_db"

    echo: bool = False
    pool_size: PositiveInt = 5
    max_overflow: NonNegativeInt = 10
    pool_timeout_seconds: PositiveInt = 30
    pool_recycle_seconds: PositiveInt = 1_800
    create_tables_on_startup: bool = True

    @property
    def dsn(self) -> str:
        """DSN completo (contém a senha — nunca logar este valor)."""
        if self.url is not None:
            return self.url.get_secret_value()
        password = quote_plus(self.password.get_secret_value())
        return f"{self.driver}://{quote_plus(self.user)}:{password}@{self.host}:{self.port}/{self.name}"

    @property
    def safe_dsn(self) -> str:
        """DSN com a senha mascarada, seguro para logs."""
        if self.url is not None:
            scheme = self.url.get_secret_value().split("://", 1)[0]
            return f"{scheme}://***"
        return f"{self.driver}://{self.user}:***@{self.host}:{self.port}/{self.name}"

    @property
    def is_sqlite(self) -> bool:
        return self.dsn.startswith("sqlite")


class RiskStratificationSettings(BaseModel):
    """Faixas de probabilidade que definem a categoria de risco clínico.

    ``p < moderate_threshold``                      -> Low
    ``moderate_threshold <= p < high_threshold``    -> Moderate
    ``p >= high_threshold``                         -> High

    O padrão de ``high_threshold`` (0.50) acompanha o limiar de Youden obtido
    no treino do DIA 1 (~0.50): um paciente "High" é exatamente aquele que o
    modelo classifica como readmissão provável.
    """

    moderate_threshold: Probability = 0.30
    high_threshold: Probability = 0.50
    top_k_factors: Annotated[int, Field(ge=1, le=15)] = 5

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.moderate_threshold >= self.high_threshold:
            raise ValueError(
                "moderate_threshold deve ser estritamente menor que high_threshold "
                f"(recebido {self.moderate_threshold} >= {self.high_threshold})."
            )
        return self


class Settings(BaseSettings):
    """Configuração raiz da aplicação."""

    model_config = SettingsConfigDict(
        env_prefix="TRIAGE_",
        env_nested_delimiter="__",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Aplicação -----------------------------------------------------------
    app_name: str = "Hospital Triage AI — Readmissão em 30 dias"
    app_version: str = "1.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"]
    )

    # --- Modelo de ML --------------------------------------------------------
    model_path: Path = PROJECT_ROOT / "models" / "readmission_model.joblib"

    # --- Sub-blocos ----------------------------------------------------------
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    risk: RiskStratificationSettings = Field(default_factory=RiskStratificationSettings)

    # --- Validadores ---------------------------------------------------------
    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, value: object) -> object:
        """Aceita ``"http://a,http://b"`` além de uma lista JSON."""
        if isinstance(value, str):
            raw = value.strip()
            if raw.startswith("["):
                import json

                return json.loads(raw)
            return [origin.strip() for origin in raw.split(",") if origin.strip()]
        return value

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError(f"log_level inválido: {value!r}")
        return level

    @field_validator("model_path")
    @classmethod
    def _resolve_model_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @model_validator(mode="after")
    def _guard_production(self) -> Self:
        """Impede configurações inseguras em produção."""
        if self.environment is Environment.PRODUCTION:
            if self.debug:
                raise ValueError("debug não pode estar ativo em produção.")
            if "*" in self.cors_origins:
                raise ValueError("CORS com '*' não é permitido em produção.")
            if (
                self.database.url is None
                and self.database.password.get_secret_value() == "triage"
            ):
                raise ValueError("Defina TRIAGE_DATABASE__PASSWORD em produção.")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION


# =============================================================================
# BOUNDARY
# =============================================================================
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna a instância única de ``Settings`` (lida uma vez do ambiente)."""
    return Settings()
