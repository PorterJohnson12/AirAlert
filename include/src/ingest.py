"""
ingest.py — AirAlert data ingestion module

Fetches one day of hourly PM2.5 readings from the OpenAQ v3 API and
matching hourly weather data from the Open-Meteo historical archive API,
merges them on (location_id, timestamp), applies gap-filling per Decision 2,
attaches the per-city routing label and centroid coordinates, and writes
the result to include/data/raw/pm25_{YYYY-MM-DD}.csv.

If the OpenAQ API is unavailable for a given location, a deterministic
synthetic dataset is generated for that location so the pipeline still runs
green (per W6A1 Part 2: "Falls back to synthetic data if OpenAQ is
unavailable — your task does not need to handle this separately").

Output schema (Contract 1):
    timestamp      datetime64[ns, UTC]  — UTC hour, one row per location per hour
    location_id    int64                — OpenAQ location ID
    city           string               — one of CITY_MODEL_KEYS; drives per-city
                                          model routing in train.py / serve.py
    latitude       float64              — decimal degrees; centroid of the location
    longitude      float64              — decimal degrees; centroid of the location
    pm25           float64              — μg/m³; NaN if sensor offline that hour
    temperature    float64              — °C from Open-Meteo
    humidity       float64              — % relative humidity from Open-Meteo
    pm25_imputed   bool                 — True if pm25 was filled by interpolation

Owner: Porter Johnson
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
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
# History window ingested per run. LAG_WINDOW_HOURS=48 in transform.py means
# rows for the current day need 48 prior hours of pm25/temperature, so we
# ingest a 3-day window ending at `date` (today + 2 prior days).
HISTORY_DAYS: int = 2

# Per-city models (Decision 6). The set of valid city labels — every row
# emitted by ingest must have city in this tuple.
CITY_MODEL_KEYS: tuple[str, ...] = ("salt_lake_city", "ogden", "provo")

# location_id -> {sensor_id, lat, lon, city} lookup. Driving the multi-location
# wrapper run_all_locations(). The OpenAQ IDs below are placeholders chosen so
# the pipeline can run end-to-end via the synthetic-data fallback when real
# sensors aren't yet selected — replace with real IDs once chosen.
LOCATION_REGISTRY: dict[int, dict] = {
    8118: {
        "city": "salt_lake_city",
        "lat": 40.7608,
        "lon": -111.8910,
        "sensor_id": 8118,
    },
    7841: {
        "city": "ogden",
        "lat": 41.2230,
        "lon": -111.9738,
        "sensor_id": 7841,
    },
    8163: {
        "city": "provo",
        "lat": 40.2338,
        "lon": -111.6585,
        "sensor_id": 8163,
    },
}

CONTRACT_1_COLUMNS: list[str] = [
    DATETIME_COL,
    "location_id",
    "city",
    "latitude",
    "longitude",
    "pm25",
    "temperature",
    "humidity",
    "pm25_imputed",
]

# ── Core fetch functions ───────────────────────────────────────────────────────


def fetch_pm25(
    sensor_id: int,
    location_id: int,
    date: str,
    api_key: str,
    history_days: int = HISTORY_DAYS,
) -> pd.DataFrame:
    """Fetch hourly PM2.5 readings for one sensor from the OpenAQ v3 API.

    Fetches the window from ``date - history_days`` 00:00 UTC through ``date``
    23:59 UTC inclusive — ``history_days + 1`` days total. The extra history
    is required because Contract 2 lag features in ``transform.py`` need
    LAG_WINDOW_HOURS (=48) prior observations per row.

    Args:
        sensor_id:    OpenAQ sensor ID (the PM2.5 sensor for this location).
        location_id:  OpenAQ location ID associated with the sensor.
        date:         End-date string in YYYY-MM-DD format (UTC).
        api_key:      OpenAQ API key — passed as X-API-Key header.
        history_days: Days of history to fetch before ``date`` (default: HISTORY_DAYS).

    Returns:
        DataFrame with columns [timestamp, location_id, pm25] where timestamp
        is datetime64[ns, UTC] and one row represents one UTC hour.

    Raises:
        requests.HTTPError: If the API returns a non-2xx status.
        requests.ConnectionError: If the API host is unreachable.
        ValueError: If OpenAQ returns zero readings for the requested window.
        KeyError: If the response JSON is missing expected fields.
    """
    start_date = (
        pd.Timestamp(date) - pd.Timedelta(days=history_days)
    ).strftime("%Y-%m-%d")
    expected_rows = 24 * (history_days + 1)
    resp = requests.get(
        f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/hours",
        headers={"X-API-Key": api_key},
        params={
            "limit": expected_rows,
            "datetime_from": f"{start_date}T00:00:00Z",
            "datetime_to": f"{date}T23:59:59Z",
        },
        timeout=30,
    )
    resp.raise_for_status()

    results = resp.json()["results"]
    if not results:
        raise ValueError(
            f"OpenAQ returned 0 readings for sensor {sensor_id} "
            f"from {start_date} to {date}"
        )
    df = pd.DataFrame({
        DATETIME_COL: pd.to_datetime(
            [r["period"]["datetimeFrom"]["utc"] for r in results], utc=True
        ),
        "location_id": location_id,
        "pm25": [r["value"] for r in results],
    })
    return df


def synthesize_pm25(
    location_id: int,
    date: str,
    history_days: int = HISTORY_DAYS,
) -> pd.DataFrame:
    """Generate a deterministic synthetic PM2.5 frame for one location/window.

    Used when the real OpenAQ API is unavailable so the W6A1 pipeline still
    runs end-to-end. The seed is derived from (location_id, date) so the same
    input always produces the same synthetic output — the run is reproducible.
    Values are biased to occasionally cross UNSAFE_THRESHOLD (35.4) so both
    classes appear in training data. Generates ``24 * (history_days + 1)``
    hourly rows ending at 23:00 UTC on ``date`` so the 48-hour lag features
    in ``transform.py`` have enough history.

    Args:
        location_id:  OpenAQ location ID — used as part of the random seed.
        date:         End-date string in YYYY-MM-DD format — used as the seed.
        history_days: Days of history to generate before ``date`` (default: HISTORY_DAYS).

    Returns:
        DataFrame with columns [timestamp, location_id, pm25] matching the
        schema of fetch_pm25.
    """
    rows = 24 * (history_days + 1)
    seed = abs(hash(f"{location_id}-{date}")) % (2**32)
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=12.0, scale=6.0, size=rows).clip(min=0.5)
    # roughly 1 in 6 hours has a spike crossing the 35.4 unsafe threshold
    spike_mask = rng.random(rows) < 0.15
    base[spike_mask] = rng.uniform(36.0, 90.0, size=spike_mask.sum())
    start_ts = pd.Timestamp(date, tz="UTC") - pd.Timedelta(days=history_days)
    timestamps = pd.date_range(start=start_ts, periods=rows, freq="h", tz="UTC")
    return pd.DataFrame({
        DATETIME_COL: timestamps,
        "location_id": location_id,
        "pm25": base,
    })


def fetch_weather(
    latitude: float,
    longitude: float,
    date: str,
    history_days: int = HISTORY_DAYS,
) -> pd.DataFrame:
    """Fetch hourly temperature and humidity from the Open-Meteo historical API.

    Fetches the window from ``date - history_days`` through ``date`` inclusive,
    matching :func:`fetch_pm25` so the merge in :func:`merge_and_fill` aligns.

    Args:
        latitude:     Decimal degrees latitude of the location centroid.
        longitude:    Decimal degrees longitude of the location centroid.
        date:         End-date string in YYYY-MM-DD format (UTC).
        history_days: Days of history to fetch before ``date`` (default: HISTORY_DAYS).

    Returns:
        DataFrame with columns [timestamp, temperature, humidity] where
        timestamp is datetime64[ns, UTC] and one row represents one UTC hour.

    Raises:
        requests.HTTPError: If the API returns a non-2xx status.
    """
    start_date = (
        pd.Timestamp(date) - pd.Timedelta(days=history_days)
    ).strftime("%Y-%m-%d")
    resp = requests.get(
        OPENMETEO_BASE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
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


def synthesize_weather(date: str, history_days: int = HISTORY_DAYS) -> pd.DataFrame:
    """Generate a deterministic synthetic weather frame for one window.

    Companion to :func:`synthesize_pm25` for use when Open-Meteo is unreachable.
    Produces plausible Utah-springtime values (cool nights, mild days, ~50% RH)
    seeded by date for reproducibility. Generates ``24 * (history_days + 1)``
    hourly rows ending at 23:00 UTC on ``date`` so the merge with
    :func:`synthesize_pm25` aligns and 48-hour lag features have history.

    Args:
        date:         End-date string in YYYY-MM-DD format — used as the seed.
        history_days: Days of history to generate before ``date`` (default: HISTORY_DAYS).

    Returns:
        DataFrame with [timestamp, temperature, humidity] matching fetch_weather.
    """
    rows = 24 * (history_days + 1)
    seed = abs(hash(f"weather-{date}")) % (2**32)
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(date, tz="UTC") - pd.Timedelta(days=history_days)
    timestamps = pd.date_range(start=start_ts, periods=rows, freq="h", tz="UTC")
    hour_of_day = np.arange(rows) % 24
    # Diurnal temperature swing centered on ~10 °C
    temperature = 10.0 + 6.0 * np.sin((hour_of_day - 6) * np.pi / 12) + rng.normal(0, 1.0, rows)
    humidity = (60.0 - 0.6 * (temperature - 10.0) + rng.normal(0, 5.0, rows)).clip(20, 95)
    return pd.DataFrame({
        DATETIME_COL: timestamps,
        "temperature": temperature,
        "humidity": humidity,
    })


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
        Merged DataFrame matching Contract 1 schema (without city/lat/lon — those
        are attached by attach_coords downstream) with pm25_imputed column.

    Raises:
        ValueError: If any gap of consecutive missing pm25 values exceeds
            MAX_GAP_HOURS hours per Decision 2.
    """
    merged = pm25_df.merge(weather_df, on=DATETIME_COL, how="left")
    merged = merged.sort_values(DATETIME_COL).reset_index(drop=True)

    merged["pm25_imputed"] = merged["pm25"].isna()

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


def attach_coords(
    df: pd.DataFrame,
    city: str,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    """Attach city, latitude, and longitude columns matching Contract 1.

    Validates that ``city`` is one of ``CITY_MODEL_KEYS`` and raises before
    bad labels can leak into the Contract 1 CSV. Returns a frame in the exact
    Contract 1 column order.

    Args:
        df:        DataFrame to label — typically the output of merge_and_fill.
        city:      One of CITY_MODEL_KEYS.
        latitude:  Decimal degrees latitude.
        longitude: Decimal degrees longitude.

    Returns:
        DataFrame with columns matching CONTRACT_1_COLUMNS exactly.

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
    df["latitude"] = latitude
    df["longitude"] = longitude
    return df[CONTRACT_1_COLUMNS]


# ── Pipeline entry points ──────────────────────────────────────────────────────


def run_ingest(
    sensor_id: int,
    location_id: int,
    latitude: float,
    longitude: float,
    date: str,
    city: str,
    output_dir: str = "include/data/raw/_per_loc",
) -> str:
    """Fetch, merge, and save one day of PM2.5 + weather data for one location.

    Orchestrates fetch_pm25 (with synthetic fallback on API failure),
    fetch_weather (with synthetic fallback on API failure), merge_and_fill,
    and attach_coords, then writes the result to
    ``{output_dir}/pm25_{location_id}_{date}.csv``. Idempotent — if the output
    file already exists, returns its path without re-fetching.

    Args:
        sensor_id:   OpenAQ sensor ID for PM2.5 at this location.
        location_id: OpenAQ location ID.
        latitude:    Location centroid latitude (decimal degrees).
        longitude:   Location centroid longitude (decimal degrees).
        date:        UTC date to ingest in YYYY-MM-DD format.
        city:        Per-city model routing label; must be in CITY_MODEL_KEYS.
        output_dir:  Directory for the per-location CSV (default:
            ``include/data/raw/_per_loc``). The combined CSV is written by
            :func:`run_all_locations`.

    Returns:
        Path to the per-location CSV, as a string.

    Raises:
        ValueError: If the location has a gap of consecutive missing hours
            exceeding MAX_GAP_HOURS, or if ``city`` is not in CITY_MODEL_KEYS.

    Note:
        If ``OPENAQ_API_KEY`` is unset, the synthetic-data fallback is used
        directly without attempting the OpenAQ call (W6A1 treats a missing
        key as "API unavailable"). Open-Meteo has no key and is always tried
        first; a network failure there also falls back to synthetic.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(output_dir) / f"pm25_{location_id}_{date}.csv")
    if Path(output_path).exists():
        return output_path

    api_key = os.environ.get("OPENAQ_API_KEY")
    if not api_key:
        print(
            f"[synthetic] OPENAQ_API_KEY not set — using synthetic PM2.5 for "
            f"location {location_id}"
        )
        pm25_df = synthesize_pm25(location_id, date)
    else:
        try:
            pm25_df = fetch_pm25(sensor_id, location_id, date, api_key)
        except (requests.HTTPError, requests.ConnectionError, ValueError) as exc:
            print(f"[synthetic] OpenAQ fetch failed for location {location_id}: {exc!r}")
            pm25_df = synthesize_pm25(location_id, date)

    try:
        weather_df = fetch_weather(latitude, longitude, date)
    except (requests.HTTPError, requests.ConnectionError) as exc:
        print(f"[synthetic] Open-Meteo fetch failed for ({latitude},{longitude}): {exc!r}")
        weather_df = synthesize_weather(date)

    merged = merge_and_fill(pm25_df, weather_df)
    merged = attach_coords(merged, city=city, latitude=latitude, longitude=longitude)

    merged.to_csv(output_path, index=False)
    return output_path


def run_all_locations(
    date: str,
    output_dir: str = "include/data/raw",
) -> str:
    """Run ingest for every entry in LOCATION_REGISTRY and produce one CSV.

    Calls :func:`run_ingest` per location, concatenates the per-location frames,
    and writes a single ``pm25_{date}.csv`` matching Contract 1. Idempotent —
    if the combined CSV already exists, returns its path without re-fetching.
    This is the entry point used by the ``fetch_air_quality`` Airflow task.

    Args:
        date:       UTC date to ingest in YYYY-MM-DD format.
        output_dir: Directory for the combined CSV (default: ``include/data/raw``).

    Returns:
        Path to the combined Contract 1 CSV, as a string.

    Raises:
        ValueError: If LOCATION_REGISTRY is empty.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(output_dir) / f"pm25_{date}.csv")
    if Path(output_path).exists():
        return output_path

    if not LOCATION_REGISTRY:
        raise ValueError(
            "LOCATION_REGISTRY is empty — nothing to ingest. Add at least one "
            "(location_id -> {city, lat, lon, sensor_id}) entry in include/src/ingest.py."
        )

    frames: list[pd.DataFrame] = []
    for location_id, meta in LOCATION_REGISTRY.items():
        per_loc_path = run_ingest(
            sensor_id=int(meta["sensor_id"]),
            location_id=location_id,
            latitude=float(meta["lat"]),
            longitude=float(meta["lon"]),
            date=date,
            city=meta["city"],
        )
        frames.append(pd.read_csv(per_loc_path, parse_dates=[DATETIME_COL]))

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_path, index=False)
    return output_path


# ── Local dev entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    date = sys.argv[1] if len(sys.argv) > 1 else "2026-04-25"
    path = run_all_locations(date)
    df = pd.read_csv(path, parse_dates=[DATETIME_COL])
    print(f"Written to {path}")
    print(f"Shape: {df.shape}, cities: {sorted(df.city.unique())}")
    print(df.head(3).to_string(index=False))
    print(f"\nDtypes:\n{df.dtypes}")
