# AirAlert — Run Instructions

End-to-end guide for running the AirAlert system and reading its output. Aimed at someone who wants to take it from zero to a working prediction, plus understand what the numbers mean.

---

## What the system does

AirAlert ingests hourly PM2.5 readings (OpenAQ) and weather (Open-Meteo) for three Utah cities — Salt Lake City, Ogden, and Provo — engineers 48 hours of lag features per location, retrains a per-city LogisticRegression model that predicts whether the next hour will exceed the EPA's 35.4 μg/m³ unsafe threshold, monitors the PM2.5 distribution for drift, exposes predictions over a FastAPI endpoint, and surfaces them in a Streamlit dashboard for non-technical users.

Pipeline order: `fetch → validate → engineer → drift_check → retrain → serve → dashboard`.

---

## Prerequisites

- **Docker Desktop** running (the Astro Runtime ships as a container)
- **Astro CLI** — install per the Astronomer docs
- **Python 3.13** — used to run FastAPI and the Streamlit dashboard outside the Airflow container
- **`.env` file** at the repo root containing `OPENAQ_API_KEY=...` (optional — if missing, ingest falls back to deterministic synthetic data)
- **MLflow** is optional. Everything still works if MLflow isn't running; you just lose the experiment UI

The repo root is `c:/AirFlowProject/AirAlert`. All commands below assume that working directory.

---

## Quick start (TL;DR)

Four shells, four commands. After each shell starts, leave it running.

```powershell
# Shell 1 — Airflow
astro dev start

# Shell 2 — (optional) MLflow UI
mlflow ui

# Shell 3 — FastAPI serving
python -m uvicorn include.src.serve:app --host 127.0.0.1 --port 8000

# Shell 4 — Streamlit dashboard
streamlit run app/dashboard.py
```

Open `http://localhost:8080` (Airflow), trigger `airalert_pipeline` for any past date, wait for all five tasks to go green, then open `http://localhost:8501` (Streamlit) and pick a city.

---

## Full run procedure

### Step 1 — Run the Airflow pipeline

1. `astro dev start` — boots scheduler, api-server, dag-processor, triggerer, and postgres containers. First start downloads the image (one-time, a few minutes).
2. Open `http://localhost:8080`. Default credentials are `admin` / `admin`.
3. In the DAGs list, click `airalert_pipeline`.
4. Click the play button (top-right), choose **Trigger DAG w/ config**, and set the **logical date** to a past date such as `2026-04-25`. The pipeline only fetches dates Open-Meteo and OpenAQ have data for; if you pick a future date, OpenAQ will return no rows and the ingest will use the synthetic fallback (still produces a valid green run).
5. Watch the Grid view. Each task square turns green as it completes. Order: `fetch_air_quality → validate_schema → engineer_features → drift_check → retrain_model`.

A green run produces the following files (all relative to the repo root):

| Path | What it is |
|---|---|
| `include/data/raw/pm25_{ds}.csv` | Contract 1 — raw merged PM2.5 + weather, 9 columns, ~216 rows for a single-day trigger (3 cities × 3 days × 24 hours) |
| `include/data/raw/_per_loc/pm25_{location_id}_{date}.csv` | Per-location intermediates (idempotency cache) |
| `include/data/features/features_{ds}.csv` | Contract 2 — engineered features, 104 columns, ~72 rows after lag-drop |
| `include/models/latest_model.pkl` | `{city: LogisticRegression}` dict loaded by `serve.py` |
| `include/models/metrics_{ds}.json` | Per-city training metrics + aggregate (f1, baseline_f1, accuracy, precision, recall) |
| `include/models/reference_pm25_stats.json` | Drift reference (mean / std / n) — used by next-day `drift_check` |
| `include/models/drift_{ds}.json` | This run's drift result |

### Step 2 — (Optional) Start the MLflow tracking server

```powershell
mlflow ui
```

This opens `http://localhost:5000`. It only matters if you want to compare runs across days. If you don't start it, the DAG detects within one second that it's unreachable and skips the logging step — every task still goes green. No code changes needed either way.

Experiments to look for:
- `AirAlert-salt_lake_city`, `AirAlert-ogden`, `AirAlert-provo` — one run per retrain per city
- `AirAlert-drift` — one run per `drift_check` execution

### Step 3 — Start the FastAPI serving endpoint

```powershell
python -m uvicorn include.src.serve:app --host 127.0.0.1 --port 8000
```

At startup, `serve.py` tries to load models from MLflow's Production stage. If MLflow isn't reachable (the TCP probe fails in under one second) it falls back to `include/models/latest_model.pkl`. Either way the server should be ready within a few seconds.

Interactive API docs at `http://localhost:8000/docs` — useful for hand-testing predictions without the dashboard.

### Step 4 — Start the Streamlit dashboard

```powershell
streamlit run app/dashboard.py
```

Opens at `http://localhost:8501`. The dashboard:
1. Probes `GET /health` on FastAPI and refuses to render if it's unreachable.
2. Lets you pick a city.
3. Reads the latest features row for that city from the most recent `include/data/features/features_*.csv`.
4. Sends it to `POST /predict` and renders the result.
5. Shows a 72-hour PM2.5 trend chart from the most recent raw CSV, with the 35.4 μg/m³ unsafe threshold drawn as a reference line.

---

## How to read the results

### Airflow Grid view

Each colored square in the Grid is one task instance. Green = success, red = failed, yellow = running. To inspect any task, click its square: the bottom panel shows logs and an **XCom** tab.

### XCom — `retrain_model` (training metrics)

Click `retrain_model` → XCom tab. The `return_value` field is a JSON dict:

| Field | What it means |
|---|---|
| `f1` | F1 score on the **unsafe class** averaged across trained cities. This is the metric Decision 3 tracks. **If this drops below 0.60, a retrain should fire** next day. |
| `baseline_f1` | F1 of a "always predict safe" model. Always 0.0 by construction — the naive baseline can never identify any unsafe hour. **Your `f1` must beat this** to demonstrate the model adds value. |
| `accuracy` | Overall accuracy on the holdout split. Misleading on imbalanced data — high accuracy with low F1 means the model is just predicting "safe" most of the time. |
| `precision` | Of all rows the model predicted unsafe, the fraction that actually were. Low precision means "crying wolf." |
| `recall` | Of all actually-unsafe rows, the fraction the model caught. Low recall means missing real unsafe hours — the worse failure mode for a public-health alert. |
| `per_city` | Same five metrics keyed by city. Look here when one city's number looks off. |
| `skipped_cities` | Cities whose training split was single-class (e.g. no unsafe hours that day) and were dropped from training. |
| `trained_cities` | Cities that did train and ship in `latest_model.pkl`. |

**Example:** if you see `f1: 0.35` and `baseline_f1: 0.0`, the model is beating the baseline but is in the retrain zone (0.35 < 0.60). If `f1: 0.0`, training succeeded but the holdout had no unsafe rows so F1 is undefined — not a model failure, just a small-sample artifact.

### XCom — `drift_check` (distribution drift)

Click `drift_check` → XCom tab. The `return_value` is a path; the actual content is in `include/models/drift_{ds}.json`:

| Field | What it means |
|---|---|
| `mean_shift_sigma` | How many standard deviations the recent PM2.5 mean has moved from the saved reference. A value of `1.8` means *"today's mean is 1.8σ above the training mean"*. |
| `drifted` | `True` if `abs(mean_shift_sigma) > 2.0` (the `DRIFT_SIGMA_THRESHOLD` from INTERFACE.md). |
| `cold_start` | `True` only on the very first run, before any reference exists. Always `False` after the first successful retrain. |
| `reference_mean` / `reference_std` | The saved snapshot from the last successful retrain. `null` on cold start. |
| `recent_mean` / `recent_std` / `n_recent` | Today's stats. Useful for sanity-checking the comparison. |

**Reading it:** `mean_shift_sigma` near zero means the input distribution is steady. Values approaching ±2 are early warnings; |value|>2 sets `drifted=True`. This is the *early-warning* signal; F1<0.60 is the *act-now* signal. Either fires a retrain per Decision 3.

### MLflow UI (if running)

`http://localhost:5000`:
- **`AirAlert-{city}`** experiments → one run per retrain. Each run logs the five training metrics plus the registered model. The model registry shows version history.
- **`AirAlert-drift`** experiment → one run per `drift_check`. Logged metrics: `mean_shift_sigma`, `drifted` (0/1), `recent_mean`, `n_recent`, and `reference_mean` when not on cold start.

### Dashboard

The plain-language sentence is the primary signal. Two shapes:

- **`✅ Air quality is predicted SAFE for the next hour (model confidence: N%). No special precautions needed for outdoor activity.`**
- **`⚠️ Air quality is predicted UNSAFE for the next hour (model confidence: N%). Sensitive groups should limit time outdoors and keep windows closed.`**

The **"Raw prediction payload"** expander shows the Contract 4 JSON:
- `is_unsafe` — `0` or `1`
- `unsafe_probability` — calibrated probability that PM2.5 exceeds 35.4 μg/m³ in the next hour (LogisticRegression's sigmoid output, per Decision 7)
- `threshold_used` — always `35.4` (the EPA 24-hr standard)

The **trend chart** shows the last 72 hours of `pm25` for the selected city with the unsafe threshold as a reference line. Bars above the line are hours that were actually unsafe; bars near it are borderline. A dataset where many bars cross the line should also show a higher `unsafe_probability` from the model — if it doesn't, that's a model-quality signal worth investigating.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `validate_schema` raises `TypeError: humidity dtype 'int64', expected 'float64'` | Stale raw CSV from before the dtype fix. | Delete `include/data/raw/pm25_*.csv` and `include/data/raw/_per_loc/pm25_*.csv`, re-trigger the DAG. |
| `engineer_features` raises `ValueError("No complete rows remain")` | Not enough lag history. The fetch should be writing 3 days; check that `HISTORY_DAYS=2` in `include/src/ingest.py`. | If the constant is correct, look for an early-return in `run_all_locations` returning a stale combined CSV. |
| FastAPI hangs at startup for ~120 s | An older version of `serve.py` without the TCP probe in `_load_from_mlflow`. | Pull the latest serve.py — the probe should fail in <1s and fall back to the joblib pickle. |
| `/predict` returns `422 — Feature vector is X hours old` | The features CSV is from a historical date and you're hitting `serve.py`'s 3-hour freshness check. | The dashboard already sidesteps this by sending a wall-clock timestamp. For manual curl tests, send `"timestamp": "<now in UTC>"`. |
| `/predict` returns `503 — No model loaded for city 'ogden'` | That city's training split was single-class on the chosen date. | Re-trigger the DAG with a different date, or wait for more data to accumulate. The other cities still serve. |
| Dashboard shows *"FastAPI service is not reachable"* | The uvicorn process isn't running. | Start it in Shell 3 above and refresh the dashboard. |
| Dashboard shows *"No features CSV found"* | The Airflow pipeline hasn't produced features yet. | Trigger the DAG and wait for `engineer_features` to go green. |
| MLflow runs aren't appearing | MLflow tracking server isn't running. | Start `mlflow ui` in Shell 2. The pipeline still goes green without it — MLflow logging is best-effort. |
| `test_dag_retries` fails | The DAG's `default_args["retries"]` is set below 2 (the included Astro template test expects `>= 2`). | Already fixed in the current DAG. If you regress this in a future edit, bump it back to 2. |

---

## Shutdown

```powershell
# Ctrl-C in the Streamlit shell
# Ctrl-C in the FastAPI shell
# Ctrl-C in the MLflow shell
astro dev stop      # in any shell
```

`astro dev stop` shuts down the Airflow containers without deleting state — next `astro dev start` resumes the same DAG history and connections. Use `astro dev kill` only when you want a clean slate.
