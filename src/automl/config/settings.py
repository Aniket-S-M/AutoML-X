"""Typed project configuration for the AutoML pipeline foundation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


EnvironmentName = Literal["local", "development", "staging", "production", "test"]


class ProjectConfig(BaseModel):
    """Project identity and reproducibility settings."""

    name: str = "automl-pipeline"
    environment: EnvironmentName = "local"
    random_seed: int = Field(default=42, ge=0)


class DataConfig(BaseModel):
    """Filesystem locations and target definition for tabular datasets."""

    input_path: Path = Path("data/raw/dataset.csv")
    raw_path: Path = Path("data/raw")
    processed_path: Path = Path("data/processed")
    target_column: str = "target"
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    validation_size: float = Field(default=0.2, gt=0.0, lt=1.0)


class TrainingConfig(BaseModel):
    """Training defaults shared by future model builders."""

    random_seed: int | None = None
    metric: str = "accuracy"


class OptimizationConfig(BaseModel):
    """Optuna configuration contract; implementation comes later."""

    enabled: bool = False
    n_trials: int = Field(default=25, ge=1)
    timeout_seconds: int | None = Field(default=None, gt=0)
    direction: Literal["minimize", "maximize"] = "maximize"


class ModelConfig(BaseModel):
    """Model artifact location and naming defaults."""

    artifact_path: Path = Path("models/artifacts")
    default_model_name: str = "baseline_model"


class MonitoringConfig(BaseModel):
    """Monitoring thresholds for future drift/performance checks."""

    threshold: float = Field(default=0.1, ge=0.0)


class RegistryConfig(BaseModel):
    """Versioned model registry location."""

    path: Path = Path("models/registry")


class Settings(BaseModel):
    """Root application settings loaded from YAML with safe defaults."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)

    @property
    def project_name(self) -> str:
        return self.project.name

    @property
    def environment(self) -> EnvironmentName:
        return self.project.environment

    @property
    def random_seed(self) -> int:
        return self.training.random_seed or self.project.random_seed

    @property
    def target_column(self) -> str:
        return self.data.target_column

    @property
    def model_registry_path(self) -> Path:
        return self.registry.path

    @property
    def test_size(self) -> float:
        return self.data.test_size

    @property
    def validation_size(self) -> float:
        return self.data.validation_size


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from YAML, falling back to defaults when no file exists."""

    selected_path = config_path or os.getenv("AUTOML_CONFIG_PATH", "configs/config.yaml")
    path = Path(selected_path)

    if not path.exists():
        return Settings()

    with path.open("r", encoding="utf-8") as config_file:
        raw_config: dict[str, Any] | None = yaml.safe_load(config_file)

    return Settings(**(raw_config or {}))
