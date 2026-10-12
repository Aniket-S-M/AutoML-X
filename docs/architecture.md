# AutoML Foundation Contracts

This document records the intended architecture contracts for later milestones.
It does not describe completed ML, API, Airflow, monitoring, Docker, or CI/CD
implementations.

## Data Contract

Data source -> ingestion -> validation -> split -> preprocessing -> model-ready
data -> model training (next milestone).

Ingestion loads configured tabular CSV data and returns a pandas DataFrame. It
does not train models, split data, preprocess features, impute values, or modify
the dataset silently.

Validation checks data quality and schema after ingestion. It reports warnings
for conditions that should be visible downstream, such as duplicate rows or
missing feature values, and raises domain errors for invalid datasets.

Input: a pandas DataFrame.

Target: the configured target column from `Settings.data.target_column`.

Data modules are responsible for loading and validating data. Splitting happens
after validation and before preprocessing is fitted. Target selection must remain
explicit rather than hidden inside model code.

## Preprocessing Contract

Input: validated DataFrame.

Output: transformed train, validation, and test feature matrices, separated
target vectors, fitted preprocessing object, and feature metadata.

Preprocessing separates `X` from `y`, infers numerical and categorical feature
columns from the training split, imputes missing feature values, scales
numerical features, and one-hot encodes categorical features with unknown
categories ignored at transform time. The target is kept separate and is not
scaled or encoded in this milestone.

Data leakage rule: preprocessing transformers are fitted only on the training
split and reused to transform validation, test, and future inference data.

## Training Contract

Input: processed training data.

Output: a trained model and training metadata.

Training metadata should include parameters, random seed, dataset references,
and any relevant fit diagnostics.

## Evaluation Contract

Input: trained model and evaluation data.

Output: metrics dictionary.

Metric names should be stable so model comparison, registry promotion, and
reports can consume them without custom parsing.

## Optimization Contract

Input: training and validation data.

Output: best parameters and optimized model.

Optuna integration should remain behind an optimization boundary so training can
run with or without hyperparameter search.

## Model Artifact Contract

A future model artifact should contain:

- model
- preprocessing pipeline
- feature/schema information
- metrics
- parameters
- configuration snapshot
- model version
- timestamp
- metadata

Artifacts should be self-describing enough for API serving and offline
evaluation to load them without re-running training.

## Registry Contract

Registry versions should be append-only and human-readable, for example:

- `model_v001`
- `model_v002`
- `model_v003`

New versions must not overwrite prior versions. Promotion rules can be added in
a later milestone.

## Prediction Contract

FastAPI will eventually consume the registered artifact through a prediction
service. Request validation and HTTP concerns should stay in the API layer;
model loading and prediction behavior should live in reusable package code.

## Airflow Contract

Airflow will eventually orchestrate reusable ML functions. DAG files should not
contain core data, preprocessing, training, evaluation, optimization, or registry
logic.
