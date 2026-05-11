# AirAlert — Interface Contract

**Team:** Porter Johnson + Ted Roper
**Last updated:** 2026-05-04

This is a living document. You will update it as you learn more about the system. That is expected and encouraged. The rule: both partners must understand and agree to every change before it is committed. Any change that affects a module boundary must be reflected in the code within the same PR.

---

## Module Ownership

| Module | Owner (writes it) | Reviewer (reviews PR) |
|---|---|---|
| `include/src/ingest.py` (W6A1 task: `fetch_air_quality`) | Porter Johnson | Ted Roper |
| `include/src/transform.py` (W6A1 task: `engineer_features`) | Ted Roper | Porter Johnson |
| `include/src/train.py` (W6A1 task: `retrain_model`) | Porter Johnson | Ted Roper |
| `validate_schema` task (inline in DAG) | Ted Roper | Porter Johnson |
| `src/serve.py` | Ted Roper | Porter Johnson |
| `dags/airalert_dag.py` | Both | Both |
| `app/dashboard.py` | Both | Both |

---

## Shared Constants

These values are fixed and must be identical everywhere they appear in the codebase.

| Constant | Value | Used in |
|---|---|---|
| `UNSAFE_THRESHOLD` | `35.4` | `transform.py`, `serve.py`, `dashboard.py` |
| `MLFLOW_EXPERIMENT` | `"AirAlert"` | `train.py`, `serve.py` |
| `MODEL_NAME` | `"AirAlert"` | `train.py`, `serve.py` |
| `MLFLOW_URI` | `"http://localhost:5000"` | `train.py`, `serve.py` |
| `OPENAQ_PM25_PARAMETER_ID` | `2` | `ingest.py` |
| `OPENMETEO_BASE_URL` | `"https://archive-api.open-meteo.com/v1/archive"` | `ingest.py` |
| `DATETIME_COL` | `"timestamp"` | `ingest.py`, `transform.py` |
| `FRESHNESS_THRESHOLD_HOURS` | `3` | `serve.py` |
| `LAG_WINDOW_HOURS` | `48` | `transform.py` |
| `MIN_COVERAGE_FRACTION` | `0.5` | `transform.py`, `train.py` |
| `MAX_GAP_HOURS` | `6` | `ingest.py` |
| `CITY_MODEL_KEYS` | `("salt_lake_city", "ogden", "provo")` | `ingest.py`, `transform.py`, `train.py`, `serve.py` |

> **Timezone rule:** All datetime values in this pipeline are stored and merged in UTC. OpenAQ returns timestamps in UTC natively. Open-Meteo returns timestamps in the timezone specified by the `&timezone=` query parameter — always pass `timezone=UTC` when calling Open-Meteo so both sources align without conversion.

---

## Data Sources

These are the two external APIs that `ingest.py` calls. Both partners should understand both sources — Person A writes the calls, Person B consumes the output.

### Source 1 — OpenAQ API v3
- **Base URL:** `https://api.openaq.org/v3`
- **Auth:** API key required — pass as `X-API-Key` header. Store in `.env`, never commit.
- **Rate limits:** Yes — if multiple students hit the API simultaneously during development, expect HTTP 429 responses. Recommended fix: cache one day's raw JSON response locally under `data/mock/` and develop against that.
- **Endpoint used:** `GET /v3/sensors/{sensors_id}/hours` — returns hourly averaged PM2.5 readings
- **PM2.5 parameter ID:** `2` (use `OPENAQ_PM25_PARAMETER_ID`)
- **Key response fields used in this pipeline:**

| Field path | Maps to column | Type | Notes |
|---|---|---|---|
| `period.datetimeFrom.utc` | `timestamp` | datetime64[ns, UTC] | Truncate to hour; already UTC |
| `value` | `pm25` | float64 | μg/m³; may be `null` if sensor offline |
| `coordinates.latitude` | _(drop after merge)_ | float64 | Used only to match location |
| `coordinates.longitude` | _(drop after merge)_ | float64 | Used only to match location |

- **Location fields** (from `GET /v3/locations/{location_id}`):

| Field | Maps to column | Notes |
|---|---|---|
| `id` | `location_id` | int64; stable OpenAQ location identifier |
| `timezone` | _(internal only)_ | For reference — do not use for datetime conversion; always convert to UTC |

### Source 2 — Open-Meteo Historical Weather API
- **Base URL:** `https://archive-api.open-meteo.com/v1/archive` (use `OPENMETEO_BASE_URL`)
- **Auth:** None required — no API key, no rate limits.
- **Endpoint used:** `GET /v1/archive` with `latitude`, `longitude`, `start_date`, `end_date`, `hourly`, and `timezone=UTC`
- **Key response fields used in this pipeline:**

| Field path | Maps to column | Type | Notes |
|---|---|---|---|
| `hourly.time[i]` | `timestamp` | datetime64[ns, UTC] | ISO 8601 string; parse with `pd.to_datetime(..., utc=True)` |
| `hourly.temperature_2m[i]` | `temperature` | float64 | °C |
| `hourly.relative_humidity_2m[i]` | `humidity` | float64 | % |

### Merge rule
OpenAQ and Open-Meteo are merged inside `_validate_merge_task` on `(location_id, timestamp)`. For this to work correctly:
- Both `timestamp` columns must be `datetime64[ns, UTC]` — no mixed-timezone joins.
- OpenAQ data must be aggregated to **one row per `(location_id, hour)`** before the merge (see Decision 5). Multiple raw readings within the same hour should be averaged.
- Open-Meteo is queried at the centroid coordinates of each location — pass `coordinates.latitude` and `coordinates.longitude` from the OpenAQ locations response.
- The merge is a **left join** on the OpenAQ side — every PM2.5 row is kept; weather rows without a PM2.5 match are dropped.

---

## Design Decisions

These are the architectural questions that shape how your system is built. They are divided into two groups.

**Decisions to make in W5D4** — these six decisions must be answered before writing any code. They directly determine the shape of Contract 2. Without them, your feature column table cannot be completed and neither partner can build a mock CSV or start coding independently.

**Decisions to defer** — these two decisions emerge naturally as you implement specific modules. Leave them blank for now and return to them at the indicated week.

For every decision you make: write your answer in one clear sentence, then explain your reasoning in 2–4 sentences that describe the tradeoff you considered and why your choice fits your system. There is no single right answer — the reasoning is what matters.

---

## Decisions to make in W5D4

### Decision 1 — Data freshness
*⚠️ Complete in W5D4 — blocks Contract 1*

**The question:** Does this pipeline need to distinguish between fresh and stale sensor data? If yes — where does that distinction matter: training, serving, or both?

**Your decision:**
Yes, we distinguish fresh from stale data, with data older than 3 hours considered stale at serving time; training uses all historical data regardless of age.

**Your reasoning:**
The training pipeline fetches a full day of historical readings where all data is already hours old — applying a freshness threshold there would arbitrarily discard valid training examples without benefit. The serving layer, however, responds to real-time queries where a prediction made from 6-hour-old features could mislead a user about current air quality. Limiting serving inputs to the past 3 hours balances responsiveness with accuracy, and the threshold is stored as `FRESHNESS_THRESHOLD_HOURS = 3` so it can be adjusted without touching logic.

---

### Decision 2 — Missing and unreliable sensor data
*⚠️ Complete in W5D4 — blocks Contract 1*

**The question:** Some sensors go offline for stretches of time. How should the pipeline handle locations where data is missing or incomplete for a given day?

**Your decision:**
Fill missing hourly readings with linear interpolation between the nearest valid values and add a boolean `pm25_imputed` flag column to mark affected rows.

**Your reasoning:**
Dropping missing rows would break lag features — a gap in hour 10 means `pm25_lag_1h` at hour 11 silently represents two hours ago, not one. Linear interpolation preserves the temporal structure while the `pm25_imputed` flag lets downstream modules filter or weight those rows differently. We chose interpolation over forward-fill because a rising pollution event would be underrepresented by a flat fill. If a gap exceeds 6 consecutive hours the location is dropped for that day entirely, since interpolating across such a large span would introduce more noise than signal.

---

### Decision 3 — Retraining trigger
*⚠️ Complete by W6D4 — blocks train.py structure*

**The question:** Under what conditions should the pipeline retrain the model? What metric do you track, and what threshold triggers a retrain?

**Your decision:**
Retrain immediately if the previous day's F1 score on the unsafe (positive) class drops below 0.60, evaluated per city.

**Your reasoning:**
Accuracy is misleading on this dataset — unsafe hours are the minority class, so a model that always predicts "safe" can score 90%+ accuracy while being useless for the only prediction users actually care about. F1 on the unsafe class jointly captures precision (don't cry wolf) and recall (don't miss real unsafe hours), and 0.60 is the operating point below which a public-health alert system stops being trustworthy: missing too many unsafe hours undermines the entire reason the system exists. Checking against the previous day rather than a rolling window means we catch drift fast — when a sensor calibration shifts or a smoke event changes the input distribution, we want the retrain to fire the next morning, not seven days later.

---

### Decision 4 — Feature engineering choices
*⚠️ Complete in W5D4 — directly produces Contract 2*

**The question:** What features will `transform.py` produce for `train.py`? What lag windows, temporal features, and aggregations will you use?

**Your agreed feature list** (complete this before writing any code — this becomes Contract 2):

| Feature | How computed | Why it's useful |
|---|---|---|
| `is_unsafe` | `pm25 > 35.4` (EPA 24-hr standard) | Target variable for classification |
| `pm25_lag_1h` … `pm25_lag_48h` | Previous 48 hourly PM2.5 readings (rolling window) | LSTM ingests full temporal pattern; 48h captures multi-day pollution buildup |
| `temperature_lag_1h` … `temperature_lag_48h` | Previous 48 hourly temperature readings | Models how atmospheric conditions drive pollution dispersion |
| `hour_sin`, `hour_cos` | `sin(2π·hour/24)`, `cos(2π·hour/24)` | Encodes diurnal cycle continuously; avoids discontinuity at midnight |
| `day_sin`, `day_cos` | `sin(2π·dow/7)`, `cos(2π·dow/7)` | Encodes weekly activity patterns (Mon–Fri traffic vs. weekend) |

---

### Decision 5 — Aggregation granularity
*⚠️ Complete in W5D4 — blocks Contract 2*

**The question:** The OpenAQ API returns individual sensor readings — potentially multiple per location per hour. Should your pipeline work with raw readings or aggregate to one row per location per hour?

**Your decision:**
Aggregate to one row per location per hour using OpenAQ's `/hours` endpoint, which handles within-hour aggregation on the API side.

**Your reasoning:**
The `/hours` endpoint already returns pre-aggregated hourly means, making within-hour variance unavailable and the decision moot. This guarantees consistent structure (one row = one hour), makes lag features unambiguous, and aligns naturally with Open-Meteo's hourly weather data. No additional aggregation logic is needed in `transform.py`.

---

### Decision 6 — Single location vs. multi-location model
*⚠️ Complete in W5D4 — affects train.py structure and Contract 2*

**The question:** Should the model be trained on data from all locations combined (one global model) or separately per location (one model per location)?

**Your decision:**
Train one model per city — Salt Lake City, Ogden, and Provo — with each city model trained on all sensor locations within that city combined. The set of cities is locked in `CITY_MODEL_KEYS = ("salt_lake_city", "ogden", "provo")`.

**Your reasoning:**
A single global model would blur city-specific pollution patterns — Ogden's refinery corridor behaves differently than Provo near Utah Lake. Per-sensor models would have too little data per location. City-level models balance data volume with geographic specificity, keep MLflow manageable at three registered models, and make `serve.py` a simple city-keyed lookup at startup. The `city` label rides through Contract 1 from `ingest.py` (assigned via `LOCATION_REGISTRY`) and must be carried through `transform.py` so `train.py` can group rows by city at fit time.

---

## Decisions to defer

### Decision 7 — Classifier choice and probability calibration
*⏳ Complete by W6D3 — when you study `_retrain_task` and begin `serve.py`*

**The question:** What classifier will `train.py` train, and how will `serve.py` return a meaningful `unsafe_probability`?

**Things to consider:**
- Does your chosen model's `predict_proba` output produce well-calibrated probabilities, or do they cluster near 0 and 1 in a way that would mislead a user reading a confidence score?
- What does `unsafe_probability = 0.72` actually mean to someone using the dashboard — and is your model's output trustworthy enough to display that number?

**Your decision:** *(defer to W6D3)*

**Your reasoning:** *(defer to W6D3)*

---

### Decision 8 — How the dashboard sources input data
*⏳ Complete by W7D2 — when you build the dashboard*

**The question:** When a user opens the dashboard, where do the input feature values come from — manual entry, live API fetch, or pre-loaded location values?

**Things to consider:**
- Can a non-technical user reasonably be expected to know their local PM2.5 lag values, and if not, what does that mean for the usability of manual entry?
- What failure modes does each approach introduce, and which tradeoff is most acceptable given your serving architecture?

**Your decision:** *(defer to W7D2)*

**Your reasoning:** *(defer to W7D2)*

---

## Data Contracts

Complete these after your W5D4 design decisions are settled. Column names and types here must match what is actually in the code. Both partners must be able to build a mock CSV from these specs and develop their module independently.

---

### Contract 1: `ingest.py` → `transform.py`

Output file: `include/data/raw/pm25_{YYYY-MM-DD}.csv`

> This file is the merged output of both API calls. OpenAQ supplies `pm25` and `location_id`. Open-Meteo supplies `temperature` and `humidity`. They are joined on `(location_id, timestamp)` — see the merge rule in the Data Sources section above.

| Column | Source | Type | Nullable | Notes |
|---|---|---|---|---|
| `timestamp` | Both (merge key) | datetime64[ns, UTC] | No | UTC only; one row per location per hour after aggregation |
| `location_id` | OpenAQ | int64 | No | OpenAQ location ID |
| `city` | `LOCATION_REGISTRY` lookup in `ingest.py` | string | No | One of `CITY_MODEL_KEYS`; per Decision 6 this drives per-city model routing in `transform.py`, `train.py`, and `serve.py` |
| `latitude` | `LOCATION_REGISTRY` lookup | float64 | No | Decimal degrees; centroid of the OpenAQ location |
| `longitude` | `LOCATION_REGISTRY` lookup | float64 | No | Decimal degrees; centroid of the OpenAQ location |
| `pm25` | OpenAQ | float64 | Yes | μg/m³; null if sensor offline for that hour |
| `temperature` | Open-Meteo (`temperature_2m`) | float64 | Yes | °C; null if Open-Meteo had no coverage |
| `humidity` | Open-Meteo (`relative_humidity_2m`) | float64 | Yes | %; null if Open-Meteo had no coverage |
| `pm25_imputed` | Derived (Decision 2) | bool | No | True if pm25 was filled by interpolation; False otherwise |

---

### Contract 2: `transform.py` → `train.py`

Output file: `include/data/features/features_{YYYY-MM-DD}.csv`

> Each row represents one (location, hour) observation with all lag features pre-computed. Rows with fewer than 48 valid prior hours are dropped (insufficient history). The `pm25_lag_*` and `temperature_lag_*` columns expand to 48 columns each — only representative entries shown; the full set follows the same pattern.

| Column | Type | Nullable | Example |
|---|---|---|---|
| `timestamp` | datetime64[ns, UTC] | No | 2024-01-15 06:00:00+00:00 |
| `location_id` | int64 | No | 1265 |
| `city` | string | No | `salt_lake_city` |
| `is_unsafe` | int8 | No | 0 |
| `pm25_lag_1h` | float64 | No | 8.3 |
| `pm25_lag_2h` | float64 | No | 7.1 |
| `pm25_lag_3h` | float64 | No | 6.9 |
| `…pm25_lag_48h` | float64 | No | 12.4 |
| `temperature_lag_1h` | float64 | No | 14.2 |
| `temperature_lag_2h` | float64 | No | 13.8 |
| `temperature_lag_3h` | float64 | No | 13.5 |
| `…temperature_lag_48h` | float64 | No | 9.7 |
| `hour_sin` | float64 | No | 1.0 |
| `hour_cos` | float64 | No | 0.0 |
| `day_sin` | float64 | No | 0.782 |
| `day_cos` | float64 | No | 0.623 |

> **Note:** Columns `pm25_lag_4h` through `pm25_lag_47h` and `temperature_lag_4h` through `temperature_lag_47h` follow the same pattern. Total columns: 3 (id/time/city) + 1 (target) + 48 (pm25 lags) + 48 (temp lags) + 4 (cyclical encodings) = 104. `city` is carried through from Contract 1 (Decision 6) so `train.py` can route rows to the right per-city model without a separate lookup.

---

### Contract 3: `train.py` → `serve.py`

Model registered in MLflow as `"AirAlert"` at `Production` stage.

Feature columns the model expects (must match Contract 2, excluding timestamp, location_id, is_unsafe):

```python
FEATURE_COLS = (
    [f"pm25_lag_{h}h" for h in range(1, 49)] +
    [f"temperature_lag_{h}h" for h in range(1, 49)] +
    ["hour_sin", "hour_cos", "day_sin", "day_cos"]
)
# Length: 100 features
```

*Complete Decision 7 (classifier choice) before finalising this contract.*

---

### Contract 4: `serve.py` → `dashboard.py`

*Complete Decision 8 (dashboard data sourcing) before finalising this contract.*

**API endpoint:** `POST http://localhost:8000/predict`

**Request body:** *(derive from Contract 3 feature columns)*
```json
{
  "location_id": 1265,
  "timestamp": "2024-01-15T06:00:00Z",
  "pm25_lags": [8.3, 7.1, 6.9, "...48 values total..."],
  "temperature_lags": [14.2, 13.8, 13.5, "...48 values total..."],
  "hour_sin": 1.0,
  "hour_cos": 0.0,
  "day_sin": 0.782,
  "day_cos": 0.623
}
```

**Response body:**
```json
{
  "is_unsafe": 0,
  "unsafe_probability": 0.12,
  "threshold_used": 35.4
}
```

**Health check:** `GET http://localhost:8000/health` → `{"status": "ok"}`

---

## Branch and Commit Conventions

| Convention | Your decision |
|---|---|
| Branch naming format | `[initials]/[feature]` → e.g. `pj/ingest-foundation`, `tr/transform-foundation` |
| Commit message format | `module: description` → e.g. `ingest: add schema validation` |
| PR review rule | Neither partner merges their own PR — the other must approve |
| Main branch protection | Direct pushes to main are not allowed |

---

## Contract Review Checklist

Before committing this file, both partners confirm:

- [x] Decisions 1–6 have answers with reasoning — not just values
- [x] Contract 2 has no blank rows
- [x] Both partners can build a mock CSV from Contract 1 and Contract 2 independently
- [x] Both partners have read every contract entry and agreed to it
- [x] Both partners understand what will break in their module if the upstream contract changes
- [x] Decisions 7 and 8 are present and marked as deferred with target weeks

---

## Mock Data

Once contracts are finalised, each partner creates a small mock CSV for the boundary they consume. Save to `data/mock/` (gitignored). Use for development and testing before the upstream module produces real data.

| Partner | File to create | Matches |
|---|---|---|
| Person A (builds `ingest.py`) | `data/mock/mock_ingest_output.csv` | Contract 1 exactly |
| Person B (builds `transform.py`) | `data/mock/mock_transform_output.csv` | Contract 2 exactly |

*Mock CSVs are saved to `data/mock/` and excluded from git via `.gitignore` (`*.csv`).*

---

## Change Log

When you update this document mid-project, record it here.

| Date | What changed | Why | Both partners agreed? |
|---|---|---|---|
| 2026-05-04 | Initial contract created; Decisions 1–6 answered; Contracts 1–3 complete | W5A1 deliverable | Yes |
| 2026-05-04 | Added `city` column to Contract 1; added `CITY_MODEL_KEYS` and `MAX_GAP_HOURS` to shared constants; locked target cities to SLC/Ogden/Provo | Decision 6 (per-city models) needed a way to route rows by city — `transform.py` and `train.py` couldn't otherwise group | PJ done; TR to mirror in `transform.py` (carry `city` into Contract 2) |
| 2026-05-07 | Decision 3 finalized: F1<0.60 on unsafe class, previous-day eval, per city | W6A1 Part 1 — switched from R² (wrong metric for binary classification) to F1 on unsafe class | PJ done; TR sign off in PR |
| 2026-05-07 | W6A1 ownership: PJ → fetch_air_quality + retrain_model (2 tasks); TR → engineer_features + validate_schema (2 tasks). PJ scaffolds initial validate_schema body inline in DAG; TR refines as documented owner. Also corrected serve.py owner to TR (was PJ in W5A1 split). | Balanced 2-2 split per W6A1 Part 2 | Yes |
| 2026-05-07 | Added `latitude` and `longitude` columns to Contract 1; migrated all data and source paths to `include/data/` and `include/src/` | W6A1 requires `include/` layout, Part 4 verifies 9 columns in raw CSV (lat/lon were referenced in merge rules but never made it into the schema) | PJ done; TR sign off in PR |
| 2026-05-10 | Added `city` column to Contract 2; ingest now fetches a 3-day window (`HISTORY_DAYS=2`) per run | `train.py` filters by city and 48h lag features need 48h of prior history — without these, every feature row was dropped and the green run failed | Yes |
