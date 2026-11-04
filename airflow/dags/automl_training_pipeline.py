"""Thin Airflow orchestration for the reusable AutoML-X package workflow."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from airflow.sdk import dag, task

from automl.orchestration import (
    apply_validation_promotion_gate,
    evaluate_and_publish_approved_model,
    ingest_dataset,
    preprocess_validated_dataset,
    train_and_select_artifact,
    validate_ingested_dataset,
)


ARTIFACT_ROOT = os.environ.get(
    "AUTOML_ARTIFACT_ROOT", "/opt/airflow/shared/artifacts"
)
CONFIG_PATH = os.environ.get(
    "AUTOML_CONFIG_PATH", "/opt/airflow/automl-x/configs/config.yaml"
)
INPUT_CSV = os.environ.get(
    "AUTOML_INPUT_CSV", "/opt/airflow/shared/data/input.csv"
)


@dag(
    dag_id="automl_x_training_pipeline",
    description="Validate, train, gate, evaluate, and publish one AutoML-X model",
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=3),
    default_args={
        "owner": "automl-x",
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
        "execution_timeout": timedelta(hours=1),
    },
    tags=["automl-x", "training"],
)
def automl_x_training_pipeline():
    """Pass only small metadata/artifact references between Airflow tasks."""

    @task
    def ingest() -> dict:
        # Each execution attempt gets an isolated run directory. A retry that
        # reaches later tasks reuses only this successful handoff's run ID.
        return ingest_dataset(
            ARTIFACT_ROOT,
            config_path=CONFIG_PATH,
            data_path=INPUT_CSV,
            run_id=uuid4().hex,
        )

    @task
    def validate(ingestion_handoff: dict) -> dict:
        return validate_ingested_dataset(ingestion_handoff)

    @task
    def preprocess(validation_handoff: dict) -> dict:
        return preprocess_validated_dataset(validation_handoff)

    @task
    def train_and_select(preprocessing_handoff: dict) -> dict:
        return train_and_select_artifact(preprocessing_handoff)

    @task
    def gate(selection_handoff: dict) -> dict:
        # A rejected gate raises; Airflow marks this task failed and skips the
        # downstream test-evaluation/publication task.
        return apply_validation_promotion_gate(selection_handoff)

    @task(retries=0)
    def evaluate_and_publish(gate_handoff: dict) -> dict:
        return evaluate_and_publish_approved_model(gate_handoff)

    ingestion_handoff = ingest()
    validation_handoff = validate(ingestion_handoff)
    preprocessing_handoff = preprocess(validation_handoff)
    selection_handoff = train_and_select(preprocessing_handoff)
    gate_handoff = gate(selection_handoff)
    evaluate_and_publish(gate_handoff)


automl_x_training_pipeline()
