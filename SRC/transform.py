"""
transform.py — AirAlert feature engineering module

Reads the raw ingest output (Contract 1) and produces a feature matrix
(Contract 2) ready for model training. For each (location_id, timestamp),
computes 48-hour lag windows for pm25 and temperature, adds cyclical
encodings for hour-of-day and day-of-week, and labels each row as
is_unsafe based on the EPA 24-hr PM2.5 threshold.

Input schema (Contract 1):  data/raw/pm25_{YYYY-MM-DD}.csv
Output schema (Contract 2): data/features/features_{YYYY-MM-DD}.csv

Output columns:
    timestamp               datetime64[ns, UTC]
    location_id             int64
    is_unsafe               int8    — 1 if pm25 > UNSAFE_THRESHOLD, else 0
    pm25_lag_1h … _48h      float64 — 48 lag columns
    temperature_lag_1h…_48h float64 — 48 lag columns
    hour_sin, hour_cos      float64 — cyclical hour-of-day encoding
    day_sin, day_cos        float64 — cyclical day-of-week encoding

Owner: Partner
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

# ── Constants (from INTERFACE.md) ─────────────────────────────────────────────

UNSAFE_THRESHOLD: float = 35.4
LAG_WINDOW_HOURS: int = 48
DATETIME_COL: str = "timestamp"

FEATURE_COLS: list[str] = (
    [f"pm25_lag_{h}h" for h in range(1, LAG_WINDOW_HOURS + 1)] +
    [f"temperature_lag_{h}h" for h in range(1, LAG_WINDOW_HOURS + 1)] +
    ["hour_sin", "hour_cos", "day_sin", "day_cos"]
)

# ── Feature engineering functions ─────────────────────────────────────────────


def add_lag_features(
    df: pd.DataFrame,
    column: str,
    n_lags: int = LAG_WINDOW_HOURS,
) -> pd.DataFrame:
    """Add n_lags lag columns for a single source column, grouped by location.

    Lag columns are named {column}_lag_1h, {column}_lag_2h, ..., {column}_lag_{n}h.
    The shift is applied within each location_id group after sorting by timestamp.

    Args:
        df:      DataFrame containing at minimum [timestamp, location_id, {column}].
        column:  Name of the column to lag.
        n_lags:  Number of hourly lag steps to compute (default: LAG_WINDOW_HOURS).

    Returns:
        DataFrame with n_lags new columns appended.
    """
    df = df.sort_values([DATETIME_COL, "location_id"])
    for lag in range(1, n_lags + 1):
        df[f"{column}_lag_{lag}h"] = (
            df.groupby("location_id")[column].shift(lag)
        )
    return df


def add_cyclical_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos cyclical encodings for hour-of-day and day-of-week.

    Encodes hour (0–23) and day-of-week (0=Mon … 6=Sun) as paired sin/cos
    values so the model sees the circular nature of time without a
    discontinuity at midnight or the week boundary.

    Args:
        df: DataFrame containing a UTC-aware timestamp column.

    Returns:
        DataFrame with four new columns: hour_sin, hour_cos, day_sin, day_cos.
    """
    hours = df[DATETIME_COL].dt.hour
    days = df[DATETIME_COL].dt.dayofweek

    df["hour_sin"] = (hours / 24 * 2 * math.pi).apply(math.sin)
    df["hour_cos"] = (hours / 24 * 2 * math.pi).apply(math.cos)
    df["day_sin"] = (days / 7 * 2 * math.pi).apply(math.sin)
    df["day_cos"] = (days / 7 * 2 * math.pi).apply(math.cos)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary is_unsafe classification target.

    Labels each row 1 if pm25 exceeds the EPA 24-hr standard (35.4 μg/m³),
    0 otherwise. Uses pm25 after gap-filling (pm25_imputed rows included).

    Args:
        df: DataFrame containing a pm25 column.

    Returns:
        DataFrame with is_unsafe int8 column appended.
    """
    df["is_unsafe"] = (df["pm25"] > UNSAFE_THRESHOLD).astype("int8")
    return df


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where any lag feature is NaN (insufficient history).

    Rows at the start of each location's history will have NaN in lag columns
    because there are fewer than LAG_WINDOW_HOURS prior observations. These
    rows cannot be used for training and are removed here.

    Args:
        df: Feature DataFrame with lag columns present.

    Returns:
        DataFrame with all rows containing NaN lag values removed.
    """
    lag_cols = [c for c in df.columns if "_lag_" in c]
    return df.dropna(subset=lag_cols).reset_index(drop=True)


def transform(input_path: str, output_path: str) -> str:
    """Read Contract 1 CSV, engineer features, and write Contract 2 CSV.

    Full pipeline: loads raw data, adds lag features for pm25 and temperature,
    adds cyclical time encodings, labels the target, drops rows with
    insufficient lag history, and writes the result.

    Args:
        input_path:  Path to a Contract 1 CSV (data/raw/pm25_{date}.csv).
        output_path: Destination path for the Contract 2 CSV.

    Returns:
        output_path (for XCom return in the Airflow task wrapper).

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If the resulting feature DataFrame is empty.
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, parse_dates=[DATETIME_COL])
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], utc=True)

    df = add_target(df)
    df = add_lag_features(df, "pm25")
    df = add_lag_features(df, "temperature")
    df = add_cyclical_encodings(df)
    df = drop_incomplete_rows(df)

    if df.empty:
        raise ValueError(f"No complete rows remain after feature engineering for {input_path}.")

    output_cols = [DATETIME_COL, "location_id", "is_unsafe"] + FEATURE_COLS
    df = df[output_cols]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


# ── Airflow task wrapper ───────────────────────────────────────────────────────


def transform_task(**context) -> str:
    """Airflow task: engineer features for the day's raw ingest output.

    Pulls the input file path from the upstream ingest_task via XCom.
    Writes the feature CSV to data/features/features_{ds}.csv and returns
    the path for the downstream train task.

    Args:
        **context: Airflow task context dict (injected by the DAG runner).

    Returns:
        Path to the written features CSV file.
    """
    date: str = context["ds"]
    input_path: str = context["ti"].xcom_pull(task_ids="ingest_task")
    output_path: str = f"data/features/features_{date}.csv"
    return transform(input_path, output_path)


# ── Local dev entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    date = sys.argv[1] if len(sys.argv) > 1 else "2026-04-25"
    out = transform(
        input_path=f"data/raw/pm25_{date}.csv",
        output_path=f"data/features/features_{date}.csv",
    )
    df = pd.read_csv(out)
    print(f"Written to {out}")
    print(f"Shape: {df.shape}")
    print(df.head(3).to_string(index=False))
