# AirAlert — Interface Contract

**Team:** [Student A Name] + [Student B Name]
**Last updated:** [Date]

This is a living document. You will update it as you learn more about the system. That is expected and encouraged. The rule: both partners must understand and agree to every change before it is committed. Any change that affects a module boundary must be reflected in the code within the same PR.

---

## Module Ownership

|      Module     | Owner (writes it) | Reviewer (reviews PR) |
|-----------------|-------------------|-----------------------|
| `src/ingest.py` | porter   | ted |
| `src/transform.py` | ted | porter |
| `src/train.py` | porter | ted |
| `src/serve.py` | ted | porter |
| `dags/airalert_dag.py`| Both | Both |
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

> **Timezone rule:** All datetime values in this pipeline are stored and merged in UTC. OpenAQ returns timestamps in UTC natively. Open-Meteo returns timestamps in the timezone specified by the `&timezone=` query parameter — always pass `timezone=UTC` when calling Open-Meteo so both sources align without conversion. Any additional constants that emerge from your design decisions (e.g. a freshness threshold for serving, a minimum coverage threshold for training) should be added here once decided.

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

**Things to consider:**
- The training pipeline fetches yesterday's full day of readings — every reading is already hours old when processed. Does a freshness threshold add meaningful filtering, or does it just remove valid historical data?
- The serving layer responds to on-demand prediction requests — does it matter how old the input features are when someone asks "is the air safe right now?"

**Your decision:**
Yes, we do need to distinguish between fresh and stale data, with stale data starting at 3 hours old. 

**Your reasoning:**
We want to keep historical data for training to make sure the model is accurate, then make sure results/inputs are valid and pertain close to real-time by
using the fresh data.

---

### Decision 2 — Missing and unreliable sensor data
*⚠️ Complete in W5D4 — blocks Contract 1*

**The question:** Some sensors go offline for stretches of time. How should the pipeline handle locations where data is missing or incomplete for a given day?

**Things to consider:**
- If you drop rows with missing data, what happens to lag features for that location — does the next valid reading's lag_1h still represent one hour ago?
- Does it matter whether the missingness is random (battery issue) or systematic (sensor in a high-pollution area that goes offline during spikes)?

**Your decision:**
We'd fill with the average between the gap and add an imputed flag

**Your reasoning:**
We can keep the data relatively consistent and flag the data as imputed to keep clarity without having to risk using old data to fill in empty spots.

---

### Decision 3 — Retraining trigger
*⚠️ Complete by W6D4 — blocks train.py structure*

**The question:** Under what conditions should the pipeline retrain the model? What metric do you track, and what threshold triggers a retrain?

**Things to consider:**
- How would you know your model has gotten worse? Do you have access to ground truth labels on live data to compute a performance metric after deployment?
- What is the cost of a false negative (predicting safe when air is actually unsafe), and what F1 floor reflects that cost?

**Your decision:**
When the R^2 score drops by 5% 

**Your reasoning:**
It keeps compute costs down and retrains when the model needs.

---

### Decision 4 — Feature engineering choices
*⚠️ Complete in W5D4 — directly produces Contract 2*

**The question:** What features will `transform.py` produce for `train.py`? What lag windows, temporal features, and aggregations will you use?

**Things to consider:**
- What patterns in air quality would a model benefit from knowing — time of day, recent trend, weather — and which of those are actually available at prediction time without causing leakage?
- What is the minimum feature set needed to beat the naive baseline of always predicting safe?

**Your agreed feature list** (complete this before writing any code — this becomes Contract 2):

| Feature | How computed | Why it's useful |
|---|---|---|
| is_unsafe | pm25 > 35.4 | Target variable |
| pm25_sequence_48h | Last 48 hours of PM2.5 readings | LSTM ingests full temporal pattern |
| temperature_sequence_48h | Last 48 hours of temperature | Models how weather drives pollution dispersion |
| hour_of_day_encoded | One-hot or sin/cos encoding of hour | Captures diurnal pollution cycles |
| day_of_week_encoded | One-hot or sin/cos encoding of day | Models weekly traffic/activity patterns |

---

### Decision 5 — Aggregation granularity
*⚠️ Complete in W5D4 — blocks Contract 2*

**The question:** The OpenAQ API returns individual sensor readings — potentially multiple per location per hour. Should your pipeline work with raw readings or aggregate to one row per location per hour?

**Things to consider:**
- If a location has 4 readings in one hour and 1 in the next, does a "1-hour lag" computed on raw rows actually represent one hour ago?
- Does within-hour variance in PM2.5 carry predictive signal worth preserving, or is the hourly mean sufficient?

**Your decision:**
Aggregate to one row per location per hour using OpenAQ's /hours endpoint, which handles within-hour aggregation on the API side.

**Your reasoning:**
The /hours endpoint already returns pre-aggregated hourly means, making within-hour variance unavailable and the decision moot. This guarantees consistent structure (one row = one hour), makes lag features unambiguous, and aligns naturally with Open-Meteo's hourly weather data. No additional aggregation logic is needed in transform.py

---

### Decision 6 — Single location vs. multi-location model
*⚠️ Complete in W5D4 — affects train.py structure and Contract 2*

**The question:** Should the model be trained on data from all locations combined (one global model) or separately per location (one model per location)?

**Things to consider:**
- How many rows does each location contribute to your training data — is there enough per-location data to train a reliable per-location model?
- How does your choice affect `train.py`, the MLflow registry structure, and what `serve.py` needs to load at startup?

**Your decision:**
Train one model per city, with each city model trained on all sensor locations within that city combined.

**Your reasoning:**
A single global model would blur city-specific pollution patterns — Ogden's refinery corridor behaves differently than Provo near Utah Lake. Per-sensor models would have too little data per location. City-level models balance data volume with geographic specificity, keep MLflow manageable at three registered models, and make serve.py a simple city-keyed lookup at startup.

---

## Decisions to defer

### Decision 7 — Classifier choice and probability calibration
*⏳ Complete by W6D3 — when you study `_retrain_task` and begin `serve.py`*

**The question:** What classifier will `train.py` train, and how will `serve.py` return a meaningful `unsafe_probability`?

**Things to consider:**
- Does your chosen model's `predict_proba` output produce well-calibrated probabilities, or do they cluster near 0 and 1 in a way that would mislead a user reading a confidence score?
- What does `unsafe_probability = 0.72` actually mean to someone using the dashboard — and is your model's output trustworthy enough to display that number?

**Your decision:**
Use a random forest classifier and apply isotonic regression for probability calibration.

**Your reasoning:**
Random forests are robust, handle non-linear relationships, and provide feature importance insights. However, their raw probabilities can be poorly calibrated. Isotonic regression is a non-parametric calibration method that can adjust the predicted probabilities to better reflect true likelihoods, making the `unsafe_probability` more interpretable and trustworthy for end-users.

---

### Decision 8 — How the dashboard sources input data
*⏳ Complete by W7D2 — when you build the dashboard*

**The question:** When a user opens the dashboard, where do the input feature values come from — manual entry, live API fetch, or pre-loaded location values?

**Things to consider:**
- Can a non-technical user reasonably be expected to know their local PM2.5 lag values, and if not, what does that mean for the usability of manual entry?
- What failure modes does each approach introduce, and which tradeoff is most acceptable given your serving architecture?

**Your decision:**
The user will only enter their location (city), and the dashboard will fetch the latest features for that city from a new API endpoint in serve.py that queries the most recent data and runs it through the same transformations as train.py to produce a real-time prediction.

**Your reasoning:**
We want to make it as easy and straight forward for the usetr as possible. Requiring users to manually enter technical features like PM2.5 lags would be a major barrier to usability and prone to error. By building an API endpoint that serves real-time predictions based on the latest data, we can provide a seamless experience where users simply select their city and receive an up-to-date air quality forecast without needing any technical knowledge.
---

## Data Contracts

Complete these after your W5D4 design decisions are settled. Column names and types here must match what is actually in the code. Both partners must be able to build a mock CSV from these specs and develop their module independently.

---

### Contract 1: `ingest.py` → `transform.py`

Output file: `data/raw/pm25_{YYYY-MM-DD}.csv`

> This file is the merged output of both API calls. OpenAQ supplies `pm25` and `location_id`. Open-Meteo supplies `temperature` and `humidity`. They are joined on `(location_id, timestamp)` — see the merge rule in the Data Sources section above.

| Column | Source | Type | Nullable | Notes |
|---|---|---|---|---|
| `timestamp` | Both (merge key) | datetime64[ns, UTC] | No | UTC only; one row per location per hour after aggregation |
| `location_id` | OpenAQ | int64 | No | OpenAQ location ID |
| `pm25` | OpenAQ | float64 | Yes | μg/m³; null if sensor offline for that hour |
| `temperature` | Open-Meteo (`temperature_2m`) | float64 | Yes | °C; null if Open-Meteo had no coverage |
| `humidity` | Open-Meteo (`relative_humidity_2m`) | float64 | Yes | %; null if Open-Meteo had no coverage |

*Add or remove columns based on your Decision 1 and Decision 2 answers.*

---

### Contract 2: `transform.py` → `train.py`

Output file: `data/features/features_{YYYY-MM-DD}.csv`

*Copy your feature list from Decision 4 here with types and example values added. No blank rows.*

| Column | Type | Nullable | Example |
|---|---|---|---|
| timestamp | datetime64[ns, UTC] | No | 2024-01-15 06:00:00+00:00 |
| location_id | int64 | No | 1001 |
| is_unsafe | int | No | 0 |
| | | | |
| | | | |

---

### Contract 3: `train.py` → `serve.py`

Model registered in MLflow as `"AirAlert"` at `Production` stage.

Feature columns the model expects (must match Contract 2, excluding timestamp, location_id, is_unsafe):

```python
FEATURE_COLS = []  # fill this in — copy from Decision 4
```

*Complete Decision 7 (classifier choice) before finalising this contract.*

---

### Contract 4: `serve.py` → `dashboard.py`

*Complete Decision 8 (dashboard data sourcing) before finalising this contract.*

**API endpoint:** `POST http://localhost:8000/predict`

**Request body:** *(derive from Contract 3 feature columns)*

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
| Branch naming format | e.g. `[initials]/[feature]` → `jd/ingest-foundation` |
| Commit message format | e.g. `module: description` → `ingest: add schema validation` |
| PR review rule | Neither partner merges their own PR — the other must approve |
| Main branch protection | Direct pushes to main are not allowed |

---

## Contract Review Checklist

Before committing this file, both partners confirm:

- [ ] Decisions 1–6 have answers with reasoning — not just values
- [ ] Contract 2 has no blank rows
- [ ] Both partners can build a mock CSV from Contract 1 and Contract 2 independently
- [ ] Both partners have read every contract entry and agreed to it
- [ ] Both partners understand what will break in their module if the upstream contract changes
- [ ] Decisions 7 and 8 are present and marked as deferred with target weeks

---

## Mock Data

Once contracts are finalised, each partner creates a small mock CSV for the boundary they consume. Save to `data/mock/` (gitignored). Use for development and testing before the upstream module produces real data.

| Partner | File to create | Matches |
|---|---|---|
| Person A (builds `transform.py`) | `data/mock/mock_ingest_output.csv` | Contract 1 exactly |
| Person B (builds `train.py`) | `data/mock/mock_transform_output.csv` | Contract 2 exactly |

---

## Change Log

When you update this document mid-project, record it here.

| Date | What changed | Why | Both partners agreed? |
|---|---|---|---|
| | | | |
