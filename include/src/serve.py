"""
serve.py — AirAlert model serving module

Loads one MLflow model per city at startup and exposes a FastAPI REST API
so that dashboard.py (and any other consumer) can request air-quality
predictions without touching MLflow directly.

Input (Contract 3):  feature vector matching FEATURE_COLS (100 columns)
Output (Contract 4): {"is_unsafe": int, "unsafe_probability": float,
                      "threshold_used": float}

API surface:
    POST http://localhost:8000/predict  — return a prediction for one location
    GET  http://localhost:8000/health   — liveness check

Owner: Ted Roper
"""

# NOTE: The following packages are required but are not yet in requirements.txt.
# Add them before running this module:
#
#     fastapi
#     uvicorn[standard]
#     mlflow
#     joblib            # fallback path: load include/models/latest_model.pkl
#
# pandas is already used by ingest.py and transform.py; joblib ships with
# scikit-learn so it is already pulled in transitively by include/src/train.py.

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

# ── Constants (from INTERFACE.md) ─────────────────────────────────────────────

UNSAFE_THRESHOLD: float = 35.4 # for the air
FRESHNESS_THRESHOLD_HOURS: int = 3
MLFLOW_EXPERIMENT: str = "AirAlert"
MODEL_NAME: str = "AirAlert"
MLFLOW_URI: str = "http://localhost:5000"
CITY_MODEL_KEYS: tuple[str, ...] = ("salt_lake_city", "ogden", "provo")
PORT: int = 8000
LAG_WINDOW_HOURS: int = 48

# Joblib fallback: include/src/train.py always writes a {city: estimator}
# dict here, even when MLflow is unreachable.  Used by the lifespan when the
# MLflow tracking server is down so the API can still serve predictions.
MODELS_PICKLE_PATH: Path = Path("include/models/latest_model.pkl")

# location_id → city lookup.  Mirrors the LOCATION_REGISTRY in
# include/src/ingest.py (which carries extra sensor_id / lat / lon fields
# the serving layer doesn't need).  Update both files simultaneously when
# real OpenAQ IDs change.
LOCATION_REGISTRY: dict[int, str] = {
    8118: "salt_lake_city",
    7841: "ogden",
    8163: "provo",
}

# ── Feature columns (Contract 3) ──────────────────────────────────────────────

# Copied verbatim from transform.py so both files always agree on column order.
# Any change here must be mirrored in transform.py and train.py.
FEATURE_COLS: list[str] = (
    [f"pm25_lag_{h}h" for h in range(1, LAG_WINDOW_HOURS + 1)] +
    [f"temperature_lag_{h}h" for h in range(1, LAG_WINDOW_HOURS + 1)] +
    ["hour_sin", "hour_cos", "day_sin", "day_cos"]
)  # 100 features total

_log = logging.getLogger(__name__)

# ── Request / response models ──────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """Feature vector for one (location, hour) observation.

    pm25_lags and temperature_lags must each contain exactly LAG_WINDOW_HOURS
    values ordered from most-recent (lag 1) to oldest (lag 48).
    """

    location_id: int
    timestamp: str  # ISO 8601 with Z suffix, e.g. "2024-01-15T06:00:00Z"
    pm25_lags: list[float]
    temperature_lags: list[float]
    hour_sin: float
    hour_cos: float
    day_sin: float
    day_cos: float

    @field_validator("pm25_lags", "temperature_lags")
    @classmethod
    def _check_lag_length(cls, v: list[float]) -> list[float]:
        if len(v) != LAG_WINDOW_HOURS:
            raise ValueError(
                f"Expected {LAG_WINDOW_HOURS} values, got {len(v)}."
            )
        return v


class PredictResponse(BaseModel):
    """Prediction result returned to dashboard.py (Contract 4)."""

    is_unsafe: int
    unsafe_probability: float
    threshold_used: float


# ── Global model store + lifespan ─────────────────────────────────────────────

# Populated once at startup; read-only during request handling.  Holds the
# scikit-learn estimator returned by either MLflow (mlflow.sklearn flavor)
# or joblib.load() — both expose .predict() and .predict_proba().
_models: dict[str, Any] = {}


def _load_from_mlflow() -> dict[str, Any]:
    """Try to load every city model from the MLflow tracking server.

    Uses the sklearn flavor (matching include/src/train.py's logging call)
    so the returned objects expose .predict_proba() natively.  Tries the
    Production stage first per INTERFACE.md, then falls back to the
    'latest' alias since train.py registers without transitioning stages.

    Returns:
        dict mapping each city in CITY_MODEL_KEYS to a scikit-learn estimator.

    Raises:
        mlflow.exceptions.MlflowException: If any city model cannot be loaded
            under either Production or latest.  The lifespan catches this and
            falls back to the joblib pickle.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    loaded: dict[str, Any] = {}
    for city in CITY_MODEL_KEYS:
        registered = f"{MODEL_NAME}-{city}"
        try:
            uri = f"models:/{registered}/Production"
            _log.info("Loading %s from MLflow at %s", city, uri)
            loaded[city] = mlflow.sklearn.load_model(uri)  # type: ignore[attr-defined]
        except Exception:
            uri = f"models:/{registered}/latest"
            _log.info("Production stage missing — retrying %s at %s", city, uri)
            loaded[city] = mlflow.sklearn.load_model(uri)  # type: ignore[attr-defined]
    return loaded


def _load_from_pickle() -> dict[str, Any]:
    """Load per-city models from include/models/latest_model.pkl.

    train.py writes this pickle on every successful run as a {city: estimator}
    dict, regardless of MLflow availability — so it is the reliable source of
    truth when the tracking server is down.  Cities whose training split was
    single-class (e.g. all-safe) are skipped by train.py and absent from the
    bundle; this is logged as a warning rather than a fatal error so the server
    can still serve predictions for the cities that did train.

    Returns:
        dict mapping each available city to its estimator.

    Raises:
        FileNotFoundError: If the pickle does not exist — train.py has never
            run successfully and there is nothing to serve.
    """
    if not MODELS_PICKLE_PATH.exists():
        raise FileNotFoundError(
            f"{MODELS_PICKLE_PATH} not found. Run include/src/train.py before "
            "starting the server (or start the MLflow tracking server)."
        )
    bundle = joblib.load(MODELS_PICKLE_PATH)
    missing = [c for c in CITY_MODEL_KEYS if c not in bundle]
    if missing:
        _log.warning(
            "Models missing for cities %s — those locations will return 503. "
            "Re-run include/src/train.py with enough training data to fix.",
            missing,
        )
    return {city: bundle[city] for city in CITY_MODEL_KEYS if city in bundle}


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Load all per-city models before the server begins accepting requests.

    Tries MLflow first (the contract path per INTERFACE.md) and falls back to
    the joblib pickle that train.py always writes locally.  This mirrors the
    best-effort MLflow pattern in include/src/train.py: the pipeline must
    still serve predictions if the MLflow tracking server is down.

    A complete failure (both MLflow and joblib unavailable) is fatal — the
    exception propagates so the operator sees the startup failure rather
    than a server that 503s on every request.
    """
    # Decision 7 (W6A1 placeholder): include/src/train.py uses
    # LogisticRegression(class_weight="balanced") logged via
    # mlflow.sklearn.log_model().  The estimator exposes predict_proba()
    # directly, so _run_inference() can call it without the pyfunc-flavor
    # attribute forwarding workaround.  Revisit if train.py switches
    # classifiers (e.g. to a calibrated tree ensemble) before W6D3.
    try:
        _models.update(_load_from_mlflow())
        _log.info("Loaded all %d city models from MLflow.", len(_models))
    except Exception as mlflow_exc:
        _log.warning(
            "MLflow load failed (%s) — falling back to %s",
            mlflow_exc,
            MODELS_PICKLE_PATH,
        )
        _models.update(_load_from_pickle())
        _log.info("Loaded all %d city models from pickle.", len(_models))
    yield
    # No teardown — sklearn estimators hold no persistent connections.


# ── Helper functions ───────────────────────────────────────────────────────────


def _resolve_city(location_id: int) -> str:
    """Map an OpenAQ location_id to its city string.

    Args:
        location_id: OpenAQ location ID supplied in the request body.

    Returns:
        One of CITY_MODEL_KEYS identifying which per-city model to use.

    Raises:
        HTTPException(400): If location_id is not registered.
    """
    # LOCATION_REGISTRY is intentionally empty until real OpenAQ IDs are
    # chosen.  Add entries here and in ingest.py simultaneously.
    city = LOCATION_REGISTRY.get(location_id)
    if city is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown location_id {location_id}. "
                "Register it in LOCATION_REGISTRY in both include/src/serve.py "
                "and include/src/ingest.py before use."
            ),
        )
    return city


def _check_freshness(timestamp_str: str) -> datetime:
    """Parse and validate that a feature vector timestamp is not stale.

    Rejects timestamps older than FRESHNESS_THRESHOLD_HOURS (Decision 1).
    The replace("Z", "+00:00") handles the Z suffix on Python 3.10; Python
    3.11+ also parses Z natively, so this is purely defensive.

    Args:
        timestamp_str: ISO 8601 timestamp string from the request body.

    Returns:
        Parsed UTC-aware datetime.

    Raises:
        HTTPException(422): If the timestamp is more than
            FRESHNESS_THRESHOLD_HOURS hours old, or cannot be parsed.
    """
    try:
        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot parse timestamp {timestamp_str!r}: {exc}",
        ) from exc

    if ts.tzinfo is None:
        raise HTTPException(
            status_code=422,
            detail="timestamp must include a UTC offset (e.g. suffix 'Z').",
        )

    age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
    if age_hours > FRESHNESS_THRESHOLD_HOURS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Feature vector is {age_hours:.1f} hours old; "
                f"maximum allowed is {FRESHNESS_THRESHOLD_HOURS} hours "
                "(FRESHNESS_THRESHOLD_HOURS, Decision 1)."
            ),
        )
    return ts


def _build_feature_df(request: PredictRequest) -> pd.DataFrame:
    """Convert a PredictRequest into a single-row DataFrame matching FEATURE_COLS.

    Expands the pm25_lags and temperature_lags lists into named columns
    (pm25_lag_1h … pm25_lag_48h, temperature_lag_1h … temperature_lag_48h)
    and appends the four cyclical scalar features.

    Args:
        request: Validated PredictRequest with 48-element lag lists.

    Returns:
        Single-row DataFrame with columns exactly matching FEATURE_COLS in
        the correct order required by the trained model (Contract 3).
    """
    row: dict[str, float] = {}
    for h in range(1, LAG_WINDOW_HOURS + 1):
        row[f"pm25_lag_{h}h"] = request.pm25_lags[h - 1]
    for h in range(1, LAG_WINDOW_HOURS + 1):
        row[f"temperature_lag_{h}h"] = request.temperature_lags[h - 1]
    row["hour_sin"] = request.hour_sin
    row["hour_cos"] = request.hour_cos
    row["day_sin"] = request.day_sin
    row["day_cos"] = request.day_cos

    # The explicit [FEATURE_COLS] slice enforces column order, not just
    # column presence, which guards against dict insertion order drift.
    return pd.DataFrame([row])[FEATURE_COLS]


def _run_inference(
    model: Any,
    feature_df: pd.DataFrame,
) -> tuple[int, float]:
    """Run the model and return (is_unsafe, unsafe_probability).

    Args:
        model:      Loaded scikit-learn estimator for the target city
                    (LogisticRegression as of W6A1 — see Decision 7 note).
        feature_df: Single-row DataFrame from _build_feature_df().

    Returns:
        Tuple of (is_unsafe, unsafe_probability) where is_unsafe is 0 or 1
        and unsafe_probability is in [0.0, 1.0].

    Raises:
        HTTPException(503): If model inference raises an unexpected error.
    """
    # Decision 7 (W6A1 placeholder, may revisit by W6D3): train.py uses
    # LogisticRegression(class_weight="balanced") which exposes a calibrated
    # predict_proba directly.  If the team upgrades to a tree ensemble (whose
    # raw predict_proba is poorly calibrated) the fix is to wrap training in
    # CalibratedClassifierCV, NOT to add post-hoc calibration here.
    try:
        proba = model.predict_proba(feature_df)
        # Binary classifiers return [[P(safe), P(unsafe)]] — take column 1.
        unsafe_prob = float(proba[0][1])
        is_unsafe = 1 if unsafe_prob >= 0.5 else 0
        return is_unsafe, unsafe_prob
    except Exception as exc:
        _log.exception("Model inference failed: %s", exc)
        raise HTTPException(
            status_code=503, detail=f"Model inference error: {exc}"
        ) from exc


# ── FastAPI application ────────────────────────────────────────────────────────

app = FastAPI(title="AirAlert Prediction API", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return a fixed liveness response.

    Used by load balancers and Docker health checks to confirm the process
    is alive.  Does not verify that models are loaded.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Return an unsafe-air-quality prediction for one (location, hour).

    Args:
        request: Feature vector matching Contract 3, with location_id and
                 a freshness-checked timestamp.

    Returns:
        PredictResponse with is_unsafe (0/1), unsafe_probability, and
        threshold_used (always UNSAFE_THRESHOLD = 35.4 μg/m³).

    Raises:
        HTTPException(400): If location_id is not in LOCATION_REGISTRY.
        HTTPException(422): If the feature vector timestamp is stale (>3 h)
            or if Pydantic validation fails (e.g. wrong number of lags).
        HTTPException(503): If models were not loaded at startup, or if
            model inference raises an unexpected error.
    """
    if not _models:
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded. Server startup may have failed.",
        )

    city = _resolve_city(request.location_id)
    if city not in _models:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No model loaded for city '{city}'. Its training split may have "
                "been single-class — re-run the pipeline with more data."
            ),
        )
    _check_freshness(request.timestamp)
    feature_df = _build_feature_df(request)
    is_unsafe, unsafe_probability = _run_inference(_models[city], feature_df)

    return PredictResponse(
        is_unsafe=is_unsafe,
        unsafe_probability=unsafe_probability,
        threshold_used=UNSAFE_THRESHOLD,
    )


# ── Airflow task wrapper ───────────────────────────────────────────────────────


def serve_task(**context: Any) -> str:
    """Airflow task stub — serve.py is not runnable as an Airflow task.

    serve.py runs as a long-lived FastAPI process, not a finite Airflow task.
    Airflow tasks are expected to complete; a blocking uvicorn server would
    keep the task slot occupied indefinitely.

    Launch the server outside the DAG with:

        uvicorn SRC.serve:app --host 0.0.0.0 --port 8000

    If the team later decides to manage the server process from the DAG,
    replace this body with a subprocess.Popen call and return the server URL:

        import subprocess
        subprocess.Popen(["uvicorn", "SRC.serve:app", "--port", str(PORT)])
        return f"http://localhost:{PORT}/predict"

    Args:
        **context: Airflow task context dict (not used).

    Raises:
        NotImplementedError: Always — see docstring above.
    """
    raise NotImplementedError(
        "serve_task is not runnable as an Airflow task. "
        f"Launch with: uvicorn include.src.serve:app --host 0.0.0.0 --port {PORT}"
    )


# ── Local dev entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # reload=False is intentional: hot reload re-loads all per-city models on
    # every file save, which is slow.  Set reload=True only when iterating on
    # endpoint logic, not on model loading.
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("include.src.serve:app", host="0.0.0.0", port=PORT, reload=False)
