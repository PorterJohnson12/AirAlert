"""
airalert_dag.py — daily AirAlert pipeline (W6A1 + W7A1).

Wires five Airflow tasks following the TaskFlow API pattern:

    fetch_air_quality   (PJ) → include/src/ingest.py
    validate_schema     (TR) → inline @task — PJ scaffolded; TR iterates
    engineer_features   (TR) → include/src/transform.py
    drift_check         (TR) → include/src/drift.py   (added W7A1 Part 2)
    retrain_model       (PJ) → include/src/train.py

Conventions per W6A1 Part 2 (every task):
- @task decorator (no PythonOperator, no manual xcom_push/pull)
- Returns a string file path
- Uses get_current_context()['ds'] for the execution date
- Idempotency check: returns early if the output file already exists
- Raises a meaningful exception on failure
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from airflow.sdk import dag, get_current_context, task
from pendulum import datetime

# Make `include/` importable as a Python package root from inside the
# Airflow worker — Astro mounts the repo at /usr/local/airflow.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Contract 1 schema check used by validate_schema. Sourced verbatim from
# INTERFACE.md Contract 1 — keep in sync if Contract 1 changes.
CONTRACT_1_COLUMNS: list[str] = [
    "timestamp",
    "location_id",
    "city",
    "latitude",
    "longitude",
    "pm25",
    "temperature",
    "humidity",
    "pm25_imputed",
]
CONTRACT_1_DTYPES: dict[str, str] = {
    "location_id": "int64",
    "pm25": "float64",
    "temperature": "float64",
    "humidity": "float64",
    "latitude": "float64",
    "longitude": "float64",
    "pm25_imputed": "bool",
}


@dag(
    start_date=datetime(2026, 1, 1),
    schedule="0 6 * * *",  # daily at 6 AM UTC, per copilot-instructions.md
    catchup=False,
    default_args={"owner": "airalert", "retries": 2},
    tags=["airalert"],
    doc_md=__doc__,
)
def airalert_pipeline():

    @task
    def fetch_air_quality() -> str:
        """Owner: PJ. Calls OpenAQ + Open-Meteo for every entry in
        LOCATION_REGISTRY, falls back to synthetic data per location on API
        failure, writes Contract 1 CSV. Idempotent inside run_all_locations."""
        from include.src.ingest import run_all_locations

        ds = get_current_context()["ds"]
        return run_all_locations(ds)

    @task
    def validate_schema(raw_path: str) -> str:
        """Owner: TR. Pass-through validation against Contract 1.

        PJ scaffolded the initial body so the green run works on PJ's branch.
        TR refines validation rules in a follow-up PR (e.g. tighter dtype
        checks, value-range assertions).
        """
        df = pd.read_csv(raw_path, parse_dates=["timestamp"])

        missing = set(CONTRACT_1_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Contract 1 violation: missing columns {sorted(missing)} "
                f"in {raw_path}"
            )
        extra = set(df.columns) - set(CONTRACT_1_COLUMNS)
        if extra:
            raise ValueError(
                f"Contract 1 violation: unexpected columns {sorted(extra)} "
                f"in {raw_path}"
            )
        for col, expected_dtype in CONTRACT_1_DTYPES.items():
            actual_dtype = str(df[col].dtype)
            if actual_dtype != expected_dtype:
                raise TypeError(
                    f"Contract 1 violation: column {col!r} has dtype "
                    f"{actual_dtype!r}, expected {expected_dtype!r}"
                )
        return raw_path  # pass-through on success

    @task
    def engineer_features(validated_path: str) -> str:
        """Owner: TR. Reads Contract 1, produces Contract 2."""
        from include.src.transform import transform

        ds = get_current_context()["ds"]
        out = f"include/data/features/features_{ds}.csv"
        if Path(out).exists():
            return out

        Path(out).parent.mkdir(parents=True, exist_ok=True)
        return transform(validated_path, out)

    @task
    def drift_check(features_path: str) -> str:
        """Owner: TR. Compares today's PM2.5 distribution against the saved
        reference from the last successful retrain and logs mean_shift_sigma
        + drifted to MLflow under the ``AirAlert-drift`` experiment.

        Writes ``include/models/drift_{ds}.json`` and returns the path. Cold
        start (first ever run, no reference yet) is not an error: the task
        logs zero shift, sets drifted=False, and lets the DAG go green so the
        next run has a reference to compare against.
        """
        from include.src.drift import DEFAULT_REFERENCE_PATH, check_drift
        from include.src.train import MLFLOW_URI, _mlflow_reachable

        ds = get_current_context()["ds"]
        models_dir = Path("include/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        out = models_dir / f"drift_{ds}.json"
        if out.exists():
            return str(out)

        result = check_drift(features_path, reference_path=DEFAULT_REFERENCE_PATH)
        result["ds"] = ds
        result["features_path"] = features_path
        out.write_text(json.dumps(result, indent=2))

        if _mlflow_reachable(MLFLOW_URI):
            try:
                import mlflow

                mlflow.set_tracking_uri(MLFLOW_URI)
                mlflow.set_experiment("AirAlert-drift")
                with mlflow.start_run(run_name=f"drift_{ds}"):
                    mlflow.log_metric("mean_shift_sigma", result["mean_shift_sigma"])
                    mlflow.log_metric("drifted", int(result["drifted"]))
                    mlflow.log_metric("recent_mean", result["recent_mean"])
                    mlflow.log_metric("n_recent", result["n_recent"])
                    if not result["cold_start"]:
                        mlflow.log_metric("reference_mean", result["reference_mean"])
            except Exception as exc:
                print(f"[mlflow] drift logging failed: {exc!r} — continuing without it")
        else:
            print(f"[mlflow] {MLFLOW_URI} not reachable — skipping drift logging")

        return str(out)

    @task
    def retrain_model(features_path: str) -> dict:
        """Owner: PJ. Trains one model per city, logs to MLflow, writes
        latest_model.pkl + metrics_{ds}.json.

        Returns the parsed metrics dict (W6A1 Part 4: XCom return_value must
        contain f1, baseline_f1, accuracy, precision, recall). The metrics
        JSON path is included under ``metrics_path`` for downstream consumers.
        """
        from include.src.train import train

        metrics_path = train(features_path)
        metrics = json.loads(Path(metrics_path).read_text())
        metrics["metrics_path"] = metrics_path
        return metrics

    raw = fetch_air_quality()
    validated = validate_schema(raw)
    features = engineer_features(validated)
    drift_check(features) >> retrain_model(features)


airalert_pipeline()
