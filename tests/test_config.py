from pathlib import Path

import pytest
from pydantic import ValidationError

from automl.config.settings import Settings, load_settings


def test_defaults_are_available():
    settings = Settings()

    assert settings.project_name == "automl-pipeline"
    assert settings.random_seed == 42
    assert settings.target_column == "target"
    assert settings.test_size == 0.2
    assert settings.validation_size == 0.2
    assert settings.model_registry_path == Path("models/registry")


def test_config_yaml_loads():
    settings = load_settings("configs/config.yaml")

    assert settings.project.name == "automl-pipeline"
    assert settings.data.input_path == Path("data/raw/dataset.csv")
    assert settings.data.raw_path == Path("data/raw")
    assert settings.optimization.n_trials == 25
    assert settings.monitoring.threshold == 0.1


def test_missing_config_path_uses_defaults(tmp_path):
    settings = load_settings(tmp_path / "missing.yaml")

    assert settings.project_name == "automl-pipeline"
    assert settings.registry.path == Path("models/registry")


def test_invalid_split_size_is_rejected():
    with pytest.raises(ValidationError):
        Settings(data={"test_size": 1.5})
