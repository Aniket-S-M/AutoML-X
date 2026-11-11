# Running AutoML-X with Airflow

This is a local development setup using the official Apache Airflow 3.3.2
Python 3.12 image. The image installs the repository package under the matching
Airflow constraints file. Airflow's [installation guidance](https://airflow.apache.org/docs/apache-airflow/stable/installation/installing-from-pypi.html)
recommends version-specific constraints for repeatable dependency resolution.
Airflow standalone is intended for local development, not production.

## Configure the input and promotion gate

1. The project uses IBM's public [Telco Customer Churn CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv).
   To download it safely in PowerShell without overwriting an existing input:

   ```powershell
   if (Test-Path data/input.csv) { throw 'data/input.csv already exists; inspect it before replacing it' }
   Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv' -OutFile 'data/input.csv'
   ```

   On macOS/Linux, first check that `data/input.csv` does not exist, then run:

   ```bash
   test ! -e data/input.csv && curl --fail --location \
     --output data/input.csv \
     https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
   ```

   The CSV should contain 7,043 rows, 21 columns, and the `Churn` target.
   The default `configs/config.yaml` points to `data/input.csv` and sets the
   target column to `Churn`. The Airflow DAG passes `AUTOML_INPUT_CSV`, which
   Compose maps to `/opt/airflow/shared/data/input.csv`.
2. The promotion gate is configured with `metric_name: f1_yes` and
   `minimum_validation_score: 0.55`. `f1_yes` measures the `Churn=Yes` positive
   class. The gate passes when the selected model's current-run validation
   `f1_yes` is greater than or equal to `0.55`; it fails closed if either gate
   setting is missing. The `0.55` threshold is configurable and is the team's
   approved provisional value for this project. The monitoring threshold is
   not used for model promotion.
3. Keep `data/input.csv` and generated files under `artifacts/` local. The
   Compose file mounts the input read-only and the artifact directory read/write
   into all Airflow components.

Candidate selection uses validation `f1_yes`, with `Churn=Yes` as the positive
class. The project also reports positive-class precision/recall, weighted
precision/recall/F1, accuracy, ROC-AUC, PR-AUC, and a `No`/`Yes` confusion
matrix when those values are defined. The selected model's fresh validation
`f1_yes` score is checked against the configured `0.55` threshold. The untouched
test split is evaluated once only after the gate passes.

The configured `customerID` column remains in the stored source snapshot for
traceability, but is excluded from model features. Preprocessing is fitted on
the training split and then applied to the validation and test splits.

## Start Airflow

From the repository root, with Docker Desktop or Docker Engine running:

```bash
docker compose up --build
```

Open <http://localhost:8080>. Airflow standalone initializes its local
development account. Check the container logs first:

```bash
docker compose logs airflow
```

If the generated password was not printed, read Airflow's generated local
password file:

```bash
docker compose exec airflow cat /opt/airflow/simple_auth_manager_passwords.json.generated
```

Enable `automl_x_training_pipeline` in the UI, then trigger it manually. Its
schedule is explicitly disabled (`schedule=None`) because this setup expects an
operator to select/replace the input CSV and confirm the configured gate.

## Inspect runs and logs

- View task attempts and logs in the Airflow UI's DAG run and task-instance
  pages. Container logs are also available via `docker compose logs airflow`.
- Each successful ingestion creates a unique run directory under
  `artifacts/<run_id>/`. The directory contains the input snapshot, validation
  report and warnings, preprocessing and selection joblib files, and durable
  promotion-gate decision.
- A passing run publishes under
  `artifacts/<run_id>/approved/<publication_id>/`. The `manifest.json` is
  written last and identifies the inference bundle and held-out test report.
  The bundle contains the fitted preprocessor and selected fitted model.
- Gate rejection fails the gate task; the downstream final evaluation and
  publication task is skipped. Failed attempts do not modify other run
  directories. A later task retry writes atomically within the same DagRun's
  unique run directory; a retry of ingestion creates a fresh run directory.

No model registry or mutable `current` pointer is created. Treat a publication
as complete only when its manifest exists and references the bundle and test
evaluation files.

Stop the local services with:

```bash
docker compose down
```

## Validate and test

Run the package tests from the repository root:

```bash
python -m pytest -v
```

Check the Compose model and DAG Python syntax:

```bash
docker compose config --quiet
python -m compileall -q airflow/dags
```

After starting Docker, verify the DAG appears in the UI and trigger a run. To
validate the DAG with Airflow's own parser in the running image:

```bash
docker compose exec airflow airflow dags list-import-errors
```

For an end-to-end smoke run, put a CSV with a valid target column at
`data/input.csv`, verify the configured `f1_yes`/`0.55` gate, enable the DAG,
and inspect the task logs and generated manifest.
