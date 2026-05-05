"""
ingest.py — AirAlert data ingestion module

Fetches one day of hourly PM2.5 readings from the OpenAQ v3 API and
matching hourly weather data from the Open-Meteo historical archive API,
merges them on (location_id, timestamp), applies gap-filling per Decision 2,
and writes the result to data/raw/pm25_{YYYY-MM-DD}.csv.

Output schema (Contract 1):
    timestamp      datetime64[ns, UTC]  — UTC hour, one row per location per hour
    location_id    int64                — OpenAQ location ID
    pm25           float64              — μg/m³; NaN if sensor offline that hour
    temperature    float64              — °C from Open-Meteo
    humidity       float64              — % relative humidity from Open-Meteo
    pm25_imputed   bool                 — True if pm25 was filled by interpolation

Owner: Porter Johnson
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Constants (from INTERFACE.md) ─────────────────────────────────────────────

OPENAQ_BASE_URL: str = "https://api.openaq.org/v3"
OPENMETEO_BASE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
OPENAQ_PM25_PARAMETER_ID: int = 2
DATETIME_COL: str = "timestamp"
MAX_GAP_HOURS: int = 6  # drop location if gap exceeds this (Decision 2)

# ── Core fetch functions ───────────────────────────────────────────────────────


def fetch_pm25(
    sensor_id: int,
    location_id: int,
    date: str,
    api_key: str,
) -> pd.DataFrame:
    """Fetch hourly PM2.5 readings for one sensor from the OpenAQ v3 API.

    Args:
        sensor_id:   OpenAQ sensor ID (the PM2.5 sensor for this location).
        location_id: OpenAQ location ID associated with the sensor.
        date:        Date string in YYYY-MM-DD format (UTC day to fetch).
        api_key:     OpenAQ API key — passed as X-API-Key header.

    Returns:
        DataFrame with columns [timestamp, location_id, pm25] where timestamp
        is datetime64[ns, UTC] and one row represents one UTC hour.

    Raises:
        requests.HTTPError: If the API returns a non-2xx status.
        KeyError: If the response JSON is missing expected fields.
    """
    resp = requests.get(
        f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/hours",
        headers={"X-API-Key": api_key},
        params={
            "limit": 24,
            "datetime_from": f"{date}T00:00:00Z",
            "datetime_to": f"{date}T23:59:59Z",
        },
        timeout=30,
    )
    resp.raise_for_status()

    results = resp.json()["results"]
    df = pd.DataFrame({
        DATETIME_COL: pd.to_datetime(
            [r["period"]["datetimeFrom"]["utc"] for r in results], utc=True
        ),
        "location_id": location_id,
        "pm25": [r["value"] for r in results],
    })
    return df


def fetch_weather(
    latitude: float,
    longitude: float,
    date: str,
) -> pd.DataFrame:
    """Fetch hourly temperature and humidity from the Open-Meteo historical API.

    Args:
        latitude:  Decimal degrees latitude of the location centroid.
        longitude: Decimal degrees longitude of the location centroid.
        date:      Date string in YYYY-MM-DD format (UTC day to fetch).

    Returns:
        DataFrame with columns [timestamp, temperature, humidity] where
        timestamp is datetime64[ns, UTC] and one row represents one UTC hour.

    Raises:
        requests.HTTPError: If the API returns a non-2xx status.
    """
    resp = requests.get(
        OPENMETEO_BASE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": date,
            "end_date": date,
            "hourly": "temperature_2m,relative_humidity_2m",
            "timezone": "UTC",
        },
        timeout=30,
    )
    resp.raise_for_status()

    hourly = resp.json()["hourly"]
    df = pd.DataFrame({
        DATETIME_COL: pd.to_datetime(hourly["time"], utc=True),
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
    })
    return df


def merge_and_fill(
    pm25_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge PM2.5 and weather DataFrames and apply gap-filling (Decision 2).

    Performs a left join on timestamp, then linearly interpolates missing pm25
    values and marks them with pm25_imputed = True. Locations with gaps longer
    than MAX_GAP_HOURS consecutive hours are dropped entirely.

    Args:
        pm25_df:    Output of fetch_pm25 — columns [timestamp, location_id, pm25].
        weather_df: Output of fetch_weather — columns [timestamp, temperature, humidity].

    Returns:
        Merged DataFrame matching Contract 1 schema with pm25_imputed column.

    Raises:
        ValueError: If the merged DataFrame is empty after gap filtering.
    """
    merged = pm25_df.merge(weather_df, on=DATETIME_COL, how="left")
    merged = merged.sort_values(DATETIME_COL).reset_index(drop=True)

    merged["pm25_imputed"] = merged["pm25"].isna()

    # Drop location if any gap exceeds MAX_GAP_HOURS consecutive missing rows
    null_run = (
        merged["pm25"].isna()
        .groupby((~merged["pm25"].isna()).cumsum())
        .transform("sum")
    )
    if null_run.max() > MAX_GAP_HOURS:
        raise ValueError(
            f"Gap of {null_run.max()} consecutive missing hours exceeds "
            f"MAX_GAP_HOURS={MAX_GAP_HOURS}; dropping location."
        )

    merged["pm25"] = merged["pm25"].interpolate(method="linear")
    return merged[
        [DATETIME_COL, "location_id", "pm25", "temperature", "humidity", "pm25_imputed"]
    ]


def run_ingest(
    sensor_id: int,
    location_id: int,
    latitude: float,
    longitude: float,
    date: str,
    output_dir: str = "data/raw",
) -> str:
    """Fetch, merge, and save one day of PM2.5 + weather data for one location.

    Orchestrates fetch_pm25, fetch_weather, and merge_and_fill, then writes
    the result to data/raw/pm25_{date}.csv (Contract 1 output).

    Args:
        sensor_id:   OpenAQ sensor ID for PM2.5 at this location.
        location_id: OpenAQ location ID.
        latitude:    Location centroid latitude (decimal degrees).
        longitude:   Location centroid longitude (decimal degrees).
        date:        UTC date to ingest in YYYY-MM-DD format.
        output_dir:  Directory to write the output CSV (default: data/raw).

    Returns:
        Absolute path to the written CSV file.

    Raises:
        EnvironmentError: If OPENAQ_API_KEY is not set.
        requests.HTTPError: On API failure.
        ValueError: If the location has too many consecutive missing hours.
    """
    api_key = os.environ.get("OPENAQ_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAQ_API_KEY environment variable is not set.")

    pm25_df = fetch_pm25(sensor_id, location_id, date, api_key)
    weather_df = fetch_weather(latitude, longitude, date)
    merged = merge_and_fill(pm25_df, weather_df)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(output_dir) / f"pm25_{date}.csv")
    merged.to_csv(output_path, index=False)
    return output_path


# ── Airflow task wrapper ───────────────────────────────────────────────────────


def ingest_task(**context) -> str:
    """Airflow task: ingest one day of data for the configured location.

    Reads SENSOR_ID, LOCATION_ID, LAT, LON from environment variables.
    Uses context['ds'] as the target date. Returns the output file path
    via XCom for the downstream transform task.

    Args:
        **context: Airflow task context dict (injected by the DAG runner).

    Returns:
        Path to the written CSV file (data/raw/pm25_{ds}.csv).
    """
    return run_ingest(
        sensor_id=int(os.environ["SENSOR_ID"]),
        location_id=int(os.environ["LOCATION_ID"]),
        latitude=float(os.environ["LOCATION_LAT"]),
        longitude=float(os.environ["LOCATION_LON"]),
        date=context["ds"],
    )


# ── Local dev entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # Denver La Casa NCORE — run against a single day for local testing
    path = run_ingest(
        sensor_id=2270,
        location_id=1265,
        latitude=39.7794,
        longitude=-105.00523,
        date="2026-04-25",
    )
    df = pd.read_csv(path, parse_dates=[DATETIME_COL])
    print(f"Written to {path}")
    print(df.to_string(index=False))
    print(f"\nDtypes:\n{df.dtypes}")
