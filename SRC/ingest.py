"""
ingest.py — AirAlert data ingestion module

Fetches one day of hourly PM2.5 readings from the OpenAQ v3 API and
matching hourly weather data from the Open-Meteo historical archive API,
merges them on (location_id, timestamp), applies gap-filling per Decision 2,
and writes the result to data/raw/pm25_{YYYY-MM-DD}.csv.

Output schema (Contract 1):
    timestamp      datetime64[ns, UTC]  — UTC hour, one row per location per hour
    location_id    int64                — OpenAQ location ID
    city           string               — one of CITY_MODEL_KEYS; drives per-city
                                          model routing in train.py / serve.py
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

# Per-city models (Decision 6). The set of valid city labels — every row
# emitted by ingest must have city in this tuple.
CITY_MODEL_KEYS: tuple[str, ...] = ("salt_lake_city", "ogden", "provo")

# location_id -> city lookup. Fill in the OpenAQ location IDs once the team
# has chosen specific sensors for each city. ingest_task uses this to attach
# the city label without needing an extra env var per location.
LOCATION_REGISTRY: dict[int, str] = {
    # 1265: "salt_lake_city",   # TODO: replace with real SLC OpenAQ location_id
    # 0000: "ogden",            # TODO: pick an Ogden OpenAQ location_id
    # 0000: "provo",             # TODO: pick a Provo OpenAQ location_id
}

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
    if not results:
        raise ValueError(
            f"OpenAQ returned 0 readings for sensor {sensor_id} on {date}"
        )
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
    values (filling leading and trailing gaps as well) and marks the originally-
    missing rows with ``pm25_imputed = True``. If any consecutive run of missing
    pm25 values exceeds ``MAX_GAP_HOURS``, the function aborts with ValueError;
    callers handling multiple locations should catch this to skip the bad
    location rather than failing the whole run.

    Args:
        pm25_df:    Output of fetch_pm25 — columns [timestamp, location_id, pm25].
        weather_df: Output of fetch_weather — columns [timestamp, temperature, humidity].

    Returns:
        Merged DataFrame matching Contract 1 schema with pm25_imputed column.

    Raises:
        ValueError: If any gap of consecutive missing pm25 values exceeds
            MAX_GAP_HOURS hours per Decision 2.
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

    merged["pm25"] = merged["pm25"].interpolate(
        method="linear", limit_direction="both"
    )
    return merged[
        [DATETIME_COL, "location_id", "pm25", "temperature", "humidity", "pm25_imputed"]
    ]


def attach_city(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Attach the per-city model routing label as a string column.

    Validates that ``city`` is one of ``CITY_MODEL_KEYS`` and raises before
    the bad label can leak into the Contract 1 CSV.

    Args:
        df:   DataFrame to label — typically the output of merge_and_fill.
        city: One of CITY_MODEL_KEYS.

    Returns:
        DataFrame with a ``city`` column inserted after ``location_id``.

    Raises:
        ValueError: If ``city`` is not in CITY_MODEL_KEYS.
    """
    if city not in CITY_MODEL_KEYS:
        raise ValueError(
            f"city {city!r} is not in CITY_MODEL_KEYS={CITY_MODEL_KEYS}; "
            "update INTERFACE.md if a new city is being added."
        )
    df = df.copy()
    df["city"] = city
    return df[
        [DATETIME_COL, "location_id", "city", "pm25", "temperature", "humidity", "pm25_imputed"]
    ]


def run_ingest(
    sensor_id: int,
    location_id: int,
    latitude: float,
    longitude: float,
    date: str,
    city: str,
    output_dir: str = "data/raw",
) -> str:
    """Fetch, merge, and save one day of PM2.5 + weather data for one location.

    Orchestrates fetch_pm25, fetch_weather, merge_and_fill, and attach_city,
    then writes the result to data/raw/pm25_{date}.csv (Contract 1 output).

    Args:
        sensor_id:   OpenAQ sensor ID for PM2.5 at this location.
        location_id: OpenAQ location ID.
        latitude:    Location centroid latitude (decimal degrees).
        longitude:   Location centroid longitude (decimal degrees).
        date:        UTC date to ingest in YYYY-MM-DD format.
        city:        Per-city model routing label; must be in CITY_MODEL_KEYS.
        output_dir:  Directory to write the output CSV (default: data/raw).

    Returns:
        Path to the written CSV file (relative to the current working
        directory unless ``output_dir`` is absolute).

    Raises:
        EnvironmentError: If OPENAQ_API_KEY is not set.
        requests.HTTPError: On API failure.
        ValueError: If OpenAQ returns zero readings, or if the location has a
            gap of consecutive missing hours exceeding MAX_GAP_HOURS.
    """
    api_key = os.environ.get("OPENAQ_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAQ_API_KEY environment variable is not set.")

    pm25_df = fetch_pm25(sensor_id, location_id, date, api_key)
    weather_df = fetch_weather(latitude, longitude, date)
    merged = merge_and_fill(pm25_df, weather_df)
    merged = attach_city(merged, city)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(output_dir) / f"pm25_{date}.csv")
    merged.to_csv(output_path, index=False)
    return output_path


# ── Airflow task wrapper ───────────────────────────────────────────────────────


def ingest_task(**context) -> str:
    """Airflow task: ingest one day of data for the configured location.

    Reads the following environment variables (set in ``.env`` for local dev,
    or as Airflow Variables / Astro env config in production):

        OPENAQ_API_KEY  — API key for the OpenAQ v3 service
        SENSOR_ID       — int, OpenAQ sensor ID for PM2.5 at the target location
        LOCATION_ID     — int, OpenAQ location ID (must be in LOCATION_REGISTRY)
        LOCATION_LAT    — float, location centroid latitude
        LOCATION_LON    — float, location centroid longitude

    The ``city`` label is looked up from :data:`LOCATION_REGISTRY` keyed by
    ``LOCATION_ID``. Uses ``context['ds']`` as the target date and returns
    the output file path via XCom for the downstream transform task.

    Args:
        **context: Airflow task context dict (injected by the DAG runner).

    Returns:
        Path to the written CSV file (``data/raw/pm25_{ds}.csv``).

    Raises:
        KeyError: If any of the required environment variables are missing,
            or if ``LOCATION_ID`` is not registered in LOCATION_REGISTRY.
        ValueError, requests.HTTPError: Propagated from :func:`run_ingest`.
    """
    location_id = int(os.environ["LOCATION_ID"])
    if location_id not in LOCATION_REGISTRY:
        raise KeyError(
            f"LOCATION_ID={location_id} is not in LOCATION_REGISTRY; "
            "add it (with its city) in src/ingest.py before running the DAG."
        )
    return run_ingest(
        sensor_id=int(os.environ["SENSOR_ID"]),
        location_id=location_id,
        latitude=float(os.environ["LOCATION_LAT"]),
        longitude=float(os.environ["LOCATION_LON"]),
        date=context["ds"],
        city=LOCATION_REGISTRY[location_id],
    )


# ── Local dev entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    # Local-dev test — fixtures in data/mock/ are real Denver readings, but
    # we label as salt_lake_city here just to exercise the city wiring
    # end-to-end. Replace with real SLC/Ogden/Provo IDs once chosen.
    path = run_ingest(
        sensor_id=2270,
        location_id=1265,
        latitude=39.7794,
        longitude=-105.00523,
        date="2026-04-25",
        city="salt_lake_city",
    )
    df = pd.read_csv(path, parse_dates=[DATETIME_COL])
    print(f"Written to {path}")
    print(df.to_string(index=False))
    print(f"\nDtypes:\n{df.dtypes}")
