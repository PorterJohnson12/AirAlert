# AirAlert — GitHub Copilot Instructions

This file teaches Copilot the AirAlert project's architecture, schema, and conventions.
Every suggestion Copilot makes should be consistent with the contracts defined here.

---

## Project Overview

AirAlert is an Airflow-orchestrated ML pipeline that ingests hourly PM2.5 and weather data,
engineers lag features, trains a per-city classifier, and serves real-time air quality
predictions via a FastAPI endpoint with a Streamlit dashboard.

**Stack:** Python 3.11 · Apache Airflow (Astro CLI / Docker) · pandas · scikit-learn ·
MLflow · FastAPI · Streamlit · OpenAQ API v3 · Open-Meteo API

---

## Module Ownership

| Module | Owner |
|---|---|
| `src/ingest.py` | Porter Johnson |
| `src/transform.py` | Ted Roper |
| `src/train.py` | Ted Roper |
| `src/serve.py` | Porter Johnson |
| `dags/airalert_dag.py` | Both |
| `app/dashboard.py` | Both |

---

## Airflow Conventions

- Every DAG task function must accept `**context` and return a **string file path**.
- Pull upstream paths with `context['ti'].xcom_pull(task_ids='<upstream_task_id>')`.
- Never return a DataFrame from a task — save it to disk first, then return the path.
- Use `context['ds']` (execution date string `YYYY-MM-DD`) in all output file names.

```python
def my_task(**context) -> str:
    input_path: str = context['ti'].xcom_pull(task_ids='upstream_task')
    df = pd.read_csv(input_path)
    # ... process ...
    output_path = f"data/processed/{context['ds']}.csv"
    df.to_csv(output_path, index=False)
    return output_path
```

---

## Shared Constants

```python
UNSAFE_THRESHOLD          = 35.4          # μg/m³ EPA 24-hr PM2.5 standard
FRESHNESS_THRESHOLD_HOURS = 3             # serving: reject inputs older than this
LAG_WINDOW_HOURS          = 48            # number of lag columns in Contract 2
MIN_COVERAGE_FRACTION     = 0.5           # drop location if <50% hours have data
MAX_GAP_HOURS             = 6             # ingest: abort location if gap exceeds this
CITY_MODEL_KEYS           = ("salt_lake_city", "ogden", "provo")  # one model per city (Decision 6)
MLFLOW_EXPERIMENT         = "AirAlert"
MODEL_NAME                = "AirAlert"
MLFLOW_URI                = "http://localhost:5000"
OPENAQ_PM25_PARAMETER_ID  = 2
OPENMETEO_BASE_URL        = "https://archive-api.open-meteo.com/v1/archive"
DATETIME_COL              = "timestamp"
```

**Timezone rule:** All datetimes are UTC. Pass `timezone=UTC` to Open-Meteo.
Never store or merge on local-timezone timestamps.

---

## Contract 1 Schema — `ingest.py` output → `transform.py` input

File path: `data/raw/pm25_{YYYY-MM-DD}.csv`

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `timestamp` | datetime64[ns, UTC] | No | One row per location per hour |
| `location_id` | int64 | No | OpenAQ location ID |
| `city` | string | No | One of `CITY_MODEL_KEYS`; assigned in `ingest.py` via `LOCATION_REGISTRY` lookup |
| `pm25` | float64 | Yes | μg/m³; null if sensor offline |
| `temperature` | float64 | Yes | °C from Open-Meteo |
| `humidity` | float64 | Yes | % from Open-Meteo |
| `pm25_imputed` | bool | No | True if pm25 was gap-filled by interpolation |

---

## Contract 2 Schema — `transform.py` output → `train.py` input

File path: `data/features/features_{YYYY-MM-DD}.csv`

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `timestamp` | datetime64[ns, UTC] | No | |
| `location_id` | int64 | No | |
| `is_unsafe` | int8 | No | 1 if pm25 > 35.4, else 0 |
| `pm25_lag_1h` … `pm25_lag_48h` | float64 | No | 48 columns; lag in hours |
| `temperature_lag_1h` … `temperature_lag_48h` | float64 | No | 48 columns |
| `hour_sin` | float64 | No | sin(2π·hour/24) |
| `hour_cos` | float64 | No | cos(2π·hour/24) |
| `day_sin` | float64 | No | sin(2π·day_of_week/7) |
| `day_cos` | float64 | No | cos(2π·day_of_week/7) |

**FEATURE_COLS** (what train.py and serve.py use — excludes timestamp, location_id, is_unsafe):

```python
FEATURE_COLS = (
    [f"pm25_lag_{h}h" for h in range(1, 49)] +
    [f"temperature_lag_{h}h" for h in range(1, 49)] +
    ["hour_sin", "hour_cos", "day_sin", "day_cos"]
)
```

---

## Contract 3 — MLflow model

- One model registered per city in MLflow at `http://localhost:5000`, using the naming convention `"AirAlert-{city}"` for each `city` in `CITY_MODEL_KEYS` — i.e. `"AirAlert-salt_lake_city"`, `"AirAlert-ogden"`, `"AirAlert-provo"`. All three are promoted to the `Production` stage by `train.py`.
- Each model expects input with exactly the columns in `FEATURE_COLS` above
- `serve.py` does a city-keyed lookup at startup using the inbound row's `city` value

---

## Contract 4 — FastAPI endpoint

`POST http://localhost:8000/predict`

Response:
```json
{
  "is_unsafe": 0,
  "unsafe_probability": 0.12,
  "threshold_used": 35.4
}
```

Health check: `GET http://localhost:8000/health` → `{"status": "ok"}`

---

## Key Rules for Copilot Suggestions

- Column names must match the schema tables above **exactly** — no aliases or renames.
- All datetime columns must be `datetime64[ns, UTC]` — never naive or local-timezone.
- `UNSAFE_THRESHOLD = 35.4` is the only threshold — do not hardcode `35` or `36`.
- Every function must have a complete docstring and typed signatures before implementation.
- Raise exceptions on failure — never swallow errors silently.
