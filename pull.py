# OPENAQ_API_KEY="9bddb334cb07adf84762ed4e5f4fc2f926aae63f3e0839759e004cc082a1d885"
# OPENAQ_PM25_PARAMETER_ID=2
# OPENMETEO_BASE_URL="https://archive-api.open-meteo.com/v1/archive"

"""
Pulls one day of PM2.5 (OpenAQ) and weather (Open-Meteo) for Denver La Casa NCORE.
Merges on timestamp and prints the result.
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

OPENAQ_KEY      = os.environ["OPENAQ_API_KEY"]
LOCATION_ID     = 1265
SENSOR_ID       = 2270   # PM2.5 sensor at La Casa NCORE
LAT, LON        = 39.7794, -105.00523
DATE            = "2026-04-25"

# ── OpenAQ: hourly PM2.5 ──────────────────────────────────────────────────────

resp = requests.get(
    f"https://api.openaq.org/v3/sensors/{SENSOR_ID}/hours",
    headers={"X-API-Key": OPENAQ_KEY},
    params={
        "limit": 24,
        "datetime_from": f"{DATE}T00:00:00Z",
        "datetime_to":   f"{DATE}T23:59:59Z",
    },
)
resp.raise_for_status()

rows = resp.json()["results"]
pm25_df = pd.DataFrame({
    "timestamp": pd.to_datetime([r["period"]["datetimeFrom"]["utc"] for r in rows], utc=True),
    "location_id": LOCATION_ID,
    "pm25": [r["value"] for r in rows],
})

# ── Open-Meteo: hourly weather ────────────────────────────────────────────────

resp = requests.get(
    "https://archive-api.open-meteo.com/v1/archive",
    params={
        "latitude":   LAT,
        "longitude":  LON,
        "start_date": DATE,
        "end_date":   DATE,
        "hourly":     "temperature_2m,relative_humidity_2m",
        "timezone":   "UTC",
    },
)
resp.raise_for_status()

hourly = resp.json()["hourly"]
weather_df = pd.DataFrame({
    "timestamp":   pd.to_datetime(hourly["time"], utc=True),
    "temperature": hourly["temperature_2m"],
    "humidity":    hourly["relative_humidity_2m"],
})

# ── Merge ─────────────────────────────────────────────────────────────────────

merged = pm25_df.merge(weather_df, on="timestamp", how="left")

print(f"Rows: {len(merged)}")
print(merged.to_string(index=False))
print(f"\nDtypes:\n{merged.dtypes}")