"""
airalert_dag.py — daily AirAlert pipeline (W6A1).

Wires four Airflow tasks following the TaskFlow API pattern:

    fetch_air_quality   (PJ) → include/src/ingest.py
    validate_schema     (TR) → inline @task — PJ scaffolded; TR iterates
    engineer_features   (TR) → include/src/transform.py
    retrain_model       (PJ) → include/src/train.py

Conventions per W6A1 Part 2 (every task):
- @task decorator (no PythonOperator, no manual xcom_push/pull)
- Returns a string file path
- Reads the execution date via the injected ``**context`` kwarg as ``context["ds"]``
- Idempotency check: returns early if the output file already exists
- Raises a meaningful exception on failure
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from airflow.sdk import dag, task
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
    default_args={"owner": "airalert", "retries": 1},
    tags=["airalert"],
    doc_md=__doc__,
)
def airalert_pipeline():

    @task
    def fetch_air_quality(**context) -> str:
        """Owner: PJ. Calls OpenAQ + Open-Meteo for every entry in
        LOCATION_REGISTRY, falls back to synthetic data per location on API
        failure, writes Contract 1 CSV. Idempotent inside run_all_locations."""
        from include.src.ingest import run_all_locations

        return run_all_locations(context["ds"])

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
    def engineer_features(validated_path: str, **context) -> str:
        """Owner: TR. Reads Contract 1, produces Contract 2.

        Idempotency check + import path will be finalized by TR after the
        ``include/src/transform.py`` migration lands in his PR. Until then
        this body imports from the current ``src.transform`` location.
        """
        ds = context["ds"]
        out = f"include/data/features/features_{ds}.csv"
        if Path(out).exists():
            return out

        Path(out).parent.mkdir(parents=True, exist_ok=True)

        try:
            from include.src.transform import transform  # post-TR-migration import
        except ImportError:
            from src.transform import transform  # pre-migration fallback

        return transform(validated_path, out)

    @task
    def retrain_model(features_path: str) -> str:
        """Owner: PJ. Trains one model per city, logs to MLflow, writes
        latest_model.pkl + metrics_{ds}.json. Returns the metrics JSON path."""
        from include.src.train import train

        return train(features_path)

    raw = fetch_air_quality()
    validated = validate_schema(raw)
    features = engineer_features(validated)
    retrain_model(features)


airalert_pipeline()
