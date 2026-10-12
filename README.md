# AutoML-X — Automated Machine Learning Pipeline

A modular, testable Python framework for automating core machine learning workflows, from dataset ingestion and validation to preprocessing, model training, and evaluation.

**Status:** 🚧 Under Active Development

## Overview

AutoML-X aims to simplify repetitive machine learning tasks through a configurable and extensible pipeline. The project is being developed with an emphasis on clean architecture, data quality, reproducibility, and automated testing.

The current implementation provides foundational components for loading and validating tabular datasets, preprocessing numerical and categorical features, training baseline models, and evaluating model performance.

## Key Features

- **Data Ingestion:** Load tabular datasets from CSV files.
- **Data Validation:** Check datasets for common quality issues, including missing values, duplicate records, empty features, and non-finite values.
- **Preprocessing:** Build preprocessing workflows for numerical and categorical data, including missing-value handling and encoding.
- **Model Training:** Train baseline classification models, including Logistic Regression and Random Forest.
- **Evaluation:** Calculate classification metrics to support model assessment.
- **Configuration:** Manage pipeline settings through YAML configuration.
- **Logging and Error Handling:** Provide structured logging and explicit exception handling.
- **Automated Testing:** Use pytest to test data handling, configuration, preprocessing, and model components.

## Architecture

The project follows a modular Python package structure.

```text
automl-pipeline/
├── configs/
│   └── config.yaml
├── docs/
│   └── architecture.md
├── scripts/
│   ├── prepare_telco.py
│   └── preprocessing_demo.py
├── src/
│   └── automl/
│       ├── config/
│       │   └── settings.py
│       ├── data/
│       │   ├── ingestion.py
│       │   └── validation.py
│       ├── evaluation/
│       │   └── metrics.py
│       ├── models/
│       │   └── trainer.py
│       ├── preprocessing/
│       │   └── pipeline.py
│       ├── exceptions.py
│       └── logging_config.py
├── tests/
│   ├── fixtures/
│   │   └── data/
│   └── test_*.py
├── .env.example
├── .gitignore
└── pyproject.toml
```

## Technology Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Configuration:** YAML
- **Testing:** pytest
- **Development Tools:** Git and GitHub

Refer to `pyproject.toml` for the project's declared dependencies.

## Getting Started

### Prerequisites

- Python 3.10 or newer, subject to the dependency requirements in `pyproject.toml`
- Git
- pip

### 1. Clone the repository

```bash
git clone https://github.com/Aniket-S-M/AutoML-X.git
cd AutoML-X
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the project

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If the `dev` optional dependency group is not configured in `pyproject.toml`, install the project and pytest separately:

```bash
pip install -e .
pip install pytest
```

### 4. Run the tests

```bash
python -m pytest -v
```

## Configuration

Pipeline settings are maintained in:

```text
configs/config.yaml
```

Review the configuration before running scripts or integrating components into a larger workflow.

## Example Workflow

The intended workflow is:

1. Load a dataset.
2. Validate its structure and data quality.
3. Preprocess numerical and categorical features.
4. Train a baseline classification model.
5. Evaluate the model using appropriate metrics.
6. Extend the workflow with optimization, deployment, and monitoring components.

The exact execution flow depends on the current implementation and configuration. Refer to the scripts and tests for working examples.

## Development Roadmap

The following capabilities are planned or under development and should not be considered complete until implemented and tested.

- [ ] End-to-end pipeline orchestration
- [ ] Hyperparameter optimization with Optuna
- [ ] Automated model selection and comparison
- [ ] Model persistence and versioning
- [ ] REST API for model inference using FastAPI
- [ ] Experiment tracking and model registry
- [ ] Data drift and model performance monitoring
- [ ] Scheduled workflows using Apache Airflow
- [ ] Automated retraining workflows
- [ ] Docker-based deployment and CI/CD integration

## Testing and Quality

The project uses pytest to validate core components. Tests cover data ingestion, validation, configuration, preprocessing, and model functionality.

Run the full test suite before submitting changes:

```bash
python -m pytest -v
```

## Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes and add relevant tests.
4. Run the test suite.
5. Open a pull request describing the changes.

## Author

**Aniket S. M.**

- GitHub: [@Aniket-S-M](https://github.com/Aniket-S-M)

## License

No license has been specified yet. Add a `LICENSE` file before distributing the project for reuse by others.
