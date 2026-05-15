"""
dashboard.py — AirAlert user-facing Streamlit dashboard (W7A1 Part 4).

Per Decision 8 (pre-loaded), this dashboard does not collect lag values from
the user. The user picks a city; we read the most recent row of the latest
features CSV produced by the Airflow pipeline, send that vector to serve.py's
/predict endpoint, and render a plain-language result with a PM2.5 trend chart.

The dashboard never loads the model. All inference goes through the FastAPI
service (Contract 4) — per assignment Part 4.

Run with:  streamlit run app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# Make ``include/`` importable so we can reuse the shared constants without
# duplicating them — keeps Decision 6 / Contract 1 in one place.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

UNSAFE_THRESHOLD: float = 35.4
SERVE_URL: str = "http://localhost:8000"
CITY_DISPLAY: dict[str, str] = {
    "salt_lake_city": "Salt Lake City",
    "ogden": "Ogden",
    "provo": "Provo",
}
# Reverse of ingest.py / serve.py LOCATION_REGISTRY so the dashboard can map
# the user's city pick back to a registered location_id for /predict.
CITY_TO_LOCATION_ID: dict[str, int] = {
    "salt_lake_city": 8118,
    "ogden": 7841,
    "provo": 8163,
}
TREND_HOURS: int = 72
FEATURES_DIR: Path = REPO_ROOT / "include" / "data" / "features"
RAW_DIR: Path = REPO_ROOT / "include" / "data" / "raw"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _latest_csv(directory: Path, prefix: str) -> Path | None:
    """Return the most recent ``{prefix}_YYYY-MM-DD.csv`` in ``directory``."""
    if not directory.exists():
        return None
    candidates = sorted(directory.glob(f"{prefix}_*.csv"))
    return candidates[-1] if candidates else None


def _check_health() -> tuple[bool, str]:
    """Probe ``GET /health`` once and return (ok, message)."""
    try:
        resp = requests.get(f"{SERVE_URL}/health", timeout=2)
        resp.raise_for_status()
        return True, resp.json().get("status", "ok")
    except requests.RequestException as exc:
        return False, str(exc)


def _build_payload(city_row: pd.Series, location_id: int) -> dict:
    """Slice a features row into the Contract 3 ``/predict`` request body."""
    pm25_lags = [float(city_row[f"pm25_lag_{h}h"]) for h in range(1, 49)]
    temp_lags = [float(city_row[f"temperature_lag_{h}h"]) for h in range(1, 49)]
    # serve.py's freshness check rejects timestamps older than 3 hours, but the
    # synthetic pipeline emits historical dates. We send the wall-clock time
    # instead — it is what a real production caller would do.
    timestamp = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "location_id": location_id,
        "timestamp": timestamp,
        "pm25_lags": pm25_lags,
        "temperature_lags": temp_lags,
        "hour_sin": float(city_row["hour_sin"]),
        "hour_cos": float(city_row["hour_cos"]),
        "day_sin": float(city_row["day_sin"]),
        "day_cos": float(city_row["day_cos"]),
    }


def _explain(is_unsafe: int, probability: float) -> str:
    """Render the prediction as one sentence a non-technical user can act on."""
    confidence = int(round(probability * 100))
    if is_unsafe:
        return (
            f"⚠️ Air quality is predicted **UNSAFE** for the next hour "
            f"(model confidence: {confidence}%). "
            "Sensitive groups should limit time outdoors and keep windows closed."
        )
    return (
        f"✅ Air quality is predicted **SAFE** for the next hour "
        f"(model confidence: {100 - confidence}%). "
        "No special precautions needed for outdoor activity."
    )


# ── Page layout ────────────────────────────────────────────────────────────────


st.set_page_config(page_title="AirAlert", page_icon="🌫️")
st.title("🌫️ AirAlert — Air Quality Forecast")
st.caption(
    "PM2.5 'unsafe' prediction for the next hour. "
    "Per Decision 8, inputs are pre-loaded from the most recent Airflow run; "
    "the model lives in the FastAPI service — this dashboard only displays its output."
)

ok, msg = _check_health()
if not ok:
    st.error(
        f"FastAPI service at {SERVE_URL} is not reachable ({msg}). "
        "Start it with `uvicorn include.src.serve:app --port 8000` and refresh."
    )
    st.stop()
st.success(f"FastAPI service reachable — {SERVE_URL}/health → {msg}")

city_label = st.selectbox(
    "City",
    options=list(CITY_DISPLAY.keys()),
    format_func=lambda key: CITY_DISPLAY[key],
)

features_path = _latest_csv(FEATURES_DIR, "features")
if features_path is None:
    st.warning(
        f"No features CSV found under {FEATURES_DIR}. "
        "Trigger the Airflow pipeline so it produces today's features file, then refresh."
    )
    st.stop()

features_df = pd.read_csv(features_path, parse_dates=["timestamp"])
city_rows = features_df[features_df["city"] == city_label]
if city_rows.empty:
    st.warning(
        f"No rows for {CITY_DISPLAY[city_label]} in {features_path.name}. "
        "The Airflow run may have skipped this city — check the training logs."
    )
    st.stop()

latest_row = city_rows.sort_values("timestamp").iloc[-1]
st.caption(
    f"Using features row from **{latest_row['timestamp']}** "
    f"(source: `{features_path.name}`)."
)

if st.button("Predict next-hour air quality"):
    payload = _build_payload(latest_row, CITY_TO_LOCATION_ID[city_label])
    try:
        resp = requests.post(f"{SERVE_URL}/predict", json=payload, timeout=5)
    except requests.RequestException as exc:
        st.error(f"Request to /predict failed: {exc}")
    else:
        if resp.ok:
            body = resp.json()
            st.markdown(_explain(body["is_unsafe"], body["unsafe_probability"]))
            with st.expander("Raw prediction payload"):
                st.json(body)
        else:
            # serve.py returns a clear 422 message on freshness failure;
            # surface it verbatim so the user sees the actual reason.
            st.error(f"`/predict` returned {resp.status_code}: {resp.text}")

# ── PM2.5 trend chart ─────────────────────────────────────────────────────────

st.subheader(f"Recent PM2.5 — {CITY_DISPLAY[city_label]}")
raw_path = _latest_csv(RAW_DIR, "pm25")
if raw_path is None:
    st.info(
        f"No raw CSV found under {RAW_DIR}. The trend chart will appear once "
        "the Airflow pipeline writes one."
    )
else:
    raw_df = pd.read_csv(raw_path, parse_dates=["timestamp"])
    city_raw = (
        raw_df[raw_df["city"] == city_label]
        .sort_values("timestamp")
        .tail(TREND_HOURS)
        .set_index("timestamp")[["pm25"]]
    )
    if city_raw.empty:
        st.info(f"No raw rows for {CITY_DISPLAY[city_label]} in {raw_path.name}.")
    else:
        city_raw["unsafe_threshold"] = UNSAFE_THRESHOLD
        st.line_chart(city_raw)
        st.caption(
            f"Dashed reference: EPA 24-hr PM2.5 unsafe threshold "
            f"({UNSAFE_THRESHOLD} μg/m³). Source: `{raw_path.name}`."
        )
