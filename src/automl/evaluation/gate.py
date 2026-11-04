"""Validation-metric gate for deciding whether a model may be promoted."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Mapping

from automl.config.settings import EvaluationGateConfig
from automl.exceptions import ConfigurationError, TrainingError


@dataclass(frozen=True)
class PromotionGateDecision:
    """Durable, serializable summary of one validation promotion decision."""

    metric_name: str
    observed_score: float
    required_threshold: float
    passed: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        """Return the decision as a JSON-safe mapping."""

        return asdict(self)


def evaluate_promotion_gate(
    selected_validation_metrics: Mapping[str, Any] | Any,
    gate_config: EvaluationGateConfig | Mapping[str, Any] | None,
) -> PromotionGateDecision:
    """Compare one selected model's validation score to explicit gate settings.

    This function consumes validation metrics only. It has no model or test-set
    interface, so it cannot run final test evaluation as part of the decision.
    """

    config = _validated_config(gate_config)
    metric_name = config.metric_name.strip().lower()
    threshold = config.minimum_validation_score
    if not isinstance(threshold, Real) or isinstance(threshold, bool):
        raise ConfigurationError(
            "Evaluation gate minimum_validation_score must be configured as a finite number"
        )
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ConfigurationError(
            "Evaluation gate minimum_validation_score must be finite"
        )

    metrics = _metrics_mapping(selected_validation_metrics)
    if metric_name not in metrics:
        raise TrainingError(
            f"Selected model validation metrics do not contain '{metric_name}'"
        )
    score = metrics[metric_name]
    if not isinstance(score, Real) or isinstance(score, bool):
        raise TrainingError(
            f"Selected model validation metric '{metric_name}' is not a numeric score"
        )
    score = float(score)
    if not math.isfinite(score):
        raise TrainingError(
            f"Selected model validation metric '{metric_name}' must be finite"
        )

    passed = score >= threshold
    reason = (
        f"Validation {metric_name} score {score} meets minimum {threshold}"
        if passed
        else f"Validation {metric_name} score {score} is below minimum {threshold}"
    )
    return PromotionGateDecision(
        metric_name=metric_name,
        observed_score=score,
        required_threshold=threshold,
        passed=passed,
        reason=reason,
    )


def _validated_config(
    gate_config: EvaluationGateConfig | Mapping[str, Any] | None,
) -> EvaluationGateConfig:
    if gate_config is None:
        raise ConfigurationError("Evaluation gate configuration is required")
    try:
        config = (
            gate_config
            if isinstance(gate_config, EvaluationGateConfig)
            else EvaluationGateConfig.model_validate(gate_config)
        )
    except Exception as exc:
        raise ConfigurationError("Evaluation gate configuration is invalid") from exc
    if not isinstance(config.metric_name, str) or not config.metric_name.strip():
        raise ConfigurationError("Evaluation gate metric_name must be configured")
    if config.minimum_validation_score is None:
        raise ConfigurationError(
            "Evaluation gate minimum_validation_score must be configured"
        )
    return config


def _metrics_mapping(metrics: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if hasattr(metrics, "as_dict"):
        metrics = metrics.as_dict()
    if not isinstance(metrics, Mapping):
        raise TrainingError("Selected model validation metrics are missing or invalid")
    return {str(name).strip().lower(): value for name, value in metrics.items()}
