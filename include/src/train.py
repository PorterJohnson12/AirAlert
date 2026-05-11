"""
train.py — AirAlert per-city model training (W6A1 retrain_model task).

Reads Contract 2 features, trains one binary classifier per city in
CITY_MODEL_KEYS, evaluates F1 on the unsafe (positive) class, logs each run
to MLflow as ``AirAlert-{city}``, saves all models to
``include/models/latest_model.pkl`` as a ``{city: estimator}`` dict, and
writes per-city + aggregate metrics to ``include/models/metrics_{date}.json``.

Decision 3 retrain trigger: F1 on unsafe class < 0.60 on the previous day's
holdout fires a retrain — that threshold reads off ``metrics["per_city"]``.

Decision 7 (classifier choice) is still deferred — LogisticRegression is the
W6A1 placeholder because it trains in milliseconds, gives well-defined
predict_proba, and beats accuracy on highly-imbalanced data only when paired
with class_weight="balanced", which we use here.

Owner: Porter Johnson
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# ── Constants (from INTERFACE.md) ─────────────────────────────────────────────

UNSAFE_THRESHOLD: float = 35.4
LAG_WINDOW_HOURS: int = 48
DATETIME_COL: str = "timestamp"
MLFLOW_EXPERIMENT: str = "AirAlert"
MODEL_NAME: str = "AirAlert"
MLFLOW_URI: str = "http://localhost:5000"
CITY_MODEL_KEYS: tuple[str, ...] = ("salt_lake_city", "ogden", "provo")

# Feature columns the model expects — must match Contract 2 (exclude
# timestamp, location_id, city, is_unsafe).
FEATURE_COLS: list[str] = (
    [f"pm25_lag_{h}h" for h in range(1, LAG_WINDOW_HOURS + 1)]
    + [f"temperature_lag_{h}h" for h in range(1, LAG_WINDOW_HOURS + 1)]
    + ["hour_sin", "hour_cos", "day_sin", "day_cos"]
)

TARGET_COL: str = "is_unsafe"
CITY_COL: str = "city"


# ── Per-city training ─────────────────────────────────────────────────────────


def train_one_city(
    features_df: pd.DataFrame,
    city: str,
) -> tuple[Any, dict[str, float]]:
    """Train a binary classifier for a single city and compute metrics.

    Splits the city's rows chronologically (80% train / 20% holdout — no
    random shuffle, since temporal leakage would inflate scores), fits a
    LogisticRegression with ``class_weight="balanced"`` to compensate for the
    minority unsafe class, and computes the metrics required by W6A1 Part 4
    on the unsafe (positive) class.

    Args:
        features_df: Contract 2 DataFrame containing rows for at least one city.
        city:        City label to filter on; must be in CITY_MODEL_KEYS.

    Returns:
        Tuple of ``(estimator, metrics_dict)`` where metrics_dict has keys
        ``f1``, ``baseline_f1``, ``accuracy``, ``precision``, ``recall`` —
        all computed on the unsafe class on the holdout split.

    Raises:
        ValueError: If the city has too few rows to split, or if the holdout
            contains only one class (model F1 undefined).
    """
    if city not in CITY_MODEL_KEYS:
        raise ValueError(f"city {city!r} not in CITY_MODEL_KEYS={CITY_MODEL_KEYS}")

    city_rows = (
        features_df[features_df[CITY_COL] == city]
        .sort_values(DATETIME_COL)
        .reset_index(drop=True)
    )
    if len(city_rows) < 10:
        raise ValueError(
            f"city {city!r} has only {len(city_rows)} rows — need ≥10 to train"
        )

    split = int(len(city_rows) * 0.8)
    train, holdout = city_rows.iloc[:split], city_rows.iloc[split:]

    x_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    x_holdout, y_holdout = holdout[FEATURE_COLS], holdout[TARGET_COL]

    if y_train.nunique() < 2:
        raise ValueError(
            f"city {city!r} train set is single-class "
            f"(all {y_train.iloc[0]}) — cannot fit a classifier"
        )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_train)
    predictions = model.predict(x_holdout)

    # baseline = "always predict safe" → on the unsafe class, precision/recall/f1 = 0.
    metrics = {
        "f1": float(f1_score(y_holdout, predictions, pos_label=1, zero_division=0)),
        "baseline_f1": float(
            f1_score(y_holdout, [0] * len(y_holdout), pos_label=1, zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_holdout, predictions)),
        "precision": float(
            precision_score(y_holdout, predictions, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(y_holdout, predictions, pos_label=1, zero_division=0)
        ),
        "n_train": int(len(train)),
        "n_holdout": int(len(holdout)),
        "unsafe_rate_train": float(y_train.mean()),
    }
    return model, metrics


def _mlflow_reachable(uri: str, timeout: float = 1.0) -> bool:
    """Quick TCP probe — returns True only if the tracking URI accepts connections.

    Avoids the 120s default HTTP timeout when no MLflow server is running.
    """
    from urllib.parse import urlparse
    import socket

    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def log_to_mlflow(
    city: str,
    model: Any,
    metrics: dict[str, float],
    run_id_prefix: str,
) -> None:
    """Log a city's training run to MLflow if a tracking server is reachable.

    Registers the model as ``AirAlert-{city}`` and transitions it to the
    ``Production`` stage. Failures are caught and logged as warnings — the
    pipeline must still go green when MLflow isn't running, per W6A1.

    Args:
        city:           City label (one of CITY_MODEL_KEYS).
        model:          Trained scikit-learn estimator.
        metrics:        Output of :func:`train_one_city`.
        run_id_prefix:  Prefix for the MLflow run name (typically ``ds``).
    """
    if not _mlflow_reachable(MLFLOW_URI):
        print(f"[mlflow] {MLFLOW_URI} not reachable — skipping MLflow logging for {city}")
        return

    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print("[mlflow] mlflow not installed — skipping MLflow logging")
        return

    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=f"{city}_{run_id_prefix}"):
            mlflow.log_params({
                "city": city,
                "model_class": type(model).__name__,
                "feature_count": len(FEATURE_COLS),
            })
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            )
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=f"{MODEL_NAME}-{city}",
            )
        print(f"[mlflow] logged run for {city}")
    except Exception as exc:
        # Per the assignment, the green pipeline only requires the pkl on disk.
        print(f"[mlflow] logging failed for {city}: {exc!r} — continuing without it")


# ── Pipeline entry point ──────────────────────────────────────────────────────


def train(
    input_path: str,
    models_dir: Path | str = Path("include/models"),
) -> str:
    """Train per-city models, save artifacts, return the metrics JSON path.

    Idempotent — if the metrics JSON for ``input_path``'s date already exists,
    returns its path without retraining. Saves ``latest_model.pkl`` as a
    ``{city: estimator}`` dict (W6A1 Part 4 verification: loads with
    ``joblib.load()``) and writes ``metrics_{date}.json`` with per-city
    metrics plus aggregate (mean across cities) for the keys
    ``f1, baseline_f1, accuracy, precision, recall``.

    Args:
        input_path: Path to Contract 2 features CSV (output of transform.py).
        models_dir: Directory for model + metrics artifacts.

    Returns:
        Absolute (string) path to the metrics JSON. This is what the
        ``retrain_model`` Airflow task returns over XCom.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        ValueError: If no city in CITY_MODEL_KEYS has trainable data.
    """
    input_p = Path(input_path)
    if not input_p.exists():
        raise FileNotFoundError(f"Features CSV not found: {input_path}")

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Derive date from filename (features_YYYY-MM-DD.csv) for naming the metrics JSON
    stem = input_p.stem  # e.g. "features_2026-04-25"
    date = stem.split("_", 1)[1] if "_" in stem else "unknown"
    metrics_path = models_dir / f"metrics_{date}.json"
    if metrics_path.exists():
        return str(metrics_path)

    features_df = pd.read_csv(input_p, parse_dates=[DATETIME_COL])

    per_city_models: dict[str, Any] = {}
    per_city_metrics: dict[str, dict[str, float]] = {}
    skipped: list[str] = []

    for city in CITY_MODEL_KEYS:
        try:
            model, metrics = train_one_city(features_df, city)
        except ValueError as exc:
            print(f"[train] skipping {city}: {exc}")
            skipped.append(city)
            continue
        per_city_models[city] = model
        per_city_metrics[city] = metrics
        log_to_mlflow(city, model, metrics, run_id_prefix=date)

    if not per_city_models:
        raise ValueError(
            "No city had trainable data — every city was skipped. "
            f"Reasons: {skipped}"
        )

    # Aggregate (mean across cities) for the W6A1 Part 4 verification keys
    aggregate = {
        key: float(
            sum(m[key] for m in per_city_metrics.values()) / len(per_city_metrics)
        )
        for key in ("f1", "baseline_f1", "accuracy", "precision", "recall")
    }

    joblib.dump(per_city_models, models_dir / "latest_model.pkl")

    metrics_payload = {
        **aggregate,
        "per_city": per_city_metrics,
        "skipped_cities": skipped,
        "trained_cities": list(per_city_models.keys()),
        "date": date,
        "model_path": str(models_dir / "latest_model.pkl"),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    return str(metrics_path)


# ── Local dev entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    date = sys.argv[1] if len(sys.argv) > 1 else "2026-04-25"
    out = train(
        input_path=f"include/data/features/features_{date}.csv",
    )
    print(f"Metrics written to {out}")
    print(json.dumps(json.loads(Path(out).read_text()), indent=2))
