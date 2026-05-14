# Copilot Log — AirAlert Pipeline

At the end of any session where you worked on a significant file, use this
prompt in Copilot Chat to generate your log entry:

```
Summarize our conversation today into a COPILOT_LOG entry.
Include: what I was building, the key prompts I used, what you
generated, and what I changed or corrected. Keep it to 5–6 lines.
```

Paste the output into a new entry below, do a quick read to confirm it's
accurate, and commit. Both partners need at least 4 entries each by end of Week 7.

---

## Entry Format

```
## Entry [N] — [Initials] — [Date]
**Module:** which file you were working on
**Summary:** [paste Copilot-generated summary here — 5–6 lines]
```

---

## Entries

## Entry 1 — PJ — 2026-05-07
**Module:** `include/src/train.py`
**Summary:** Building `train.py` from scratch for W6A1 after the instructor pulled the "provided" version. Asked Copilot for a per-city training scaffold given Decision 3 (F1 < 0.60 on unsafe class) and Decision 6 (per-city models). Copilot suggested `RandomForestClassifier` with default hyperparameters and `train_test_split(shuffle=True)`. **Rejected both.** Decision 7 is still deferred so I scoped down to `LogisticRegression(class_weight="balanced", solver="liblinear")` — milliseconds to train, well-defined `predict_proba`, easy to swap later. The shuffle was a leakage bug — replaced with chronological 80/20 (`iloc[:split]` / `iloc[split:]`) since lag features encode time. Also caught Copilot writing `f1_score(...)` without `pos_label=1`, which silently averages — explicitly pinned to the unsafe class.

---

## Entry 2 — PJ — 2026-05-07
**Module:** `dags/airalert_dag.py` and `include/src/train.py`
**Summary:** Wiring the DAG and tightening MLflow logging. Copilot first reached for `PythonOperator` and manual `xcom_push` — **rejected** in favor of the `@task` TaskFlow API per W6A1 Part 2. For `log_to_mlflow`, the initial Copilot version did `mlflow.set_tracking_uri(...)` followed straight by `set_experiment` and `start_run` inside one big try/except. That hung my smoke test for two minutes when MLflow wasn't running because `set_experiment` retries with a 120-second HTTP default. **Modified** to add a 1-second TCP probe (`socket.create_connection`) before the mlflow imports — if the port isn't open, skip immediately. Pipeline now runs green in seconds whether or not the user has `mlflow ui` up.

---

## Entry 3 — TR — 2026-05-07
**Module:** `include/src/transform.py`
**Summary:** Building the feature engineering pipeline for W6A1. Copilot generated `df[column].shift(lag)` applied directly on the whole stacked dataframe — **rejected** because this crosses location boundaries: SLC hour-1 silently becomes Ogden hour-0's lag feature. Fixed with `df.groupby("location_id")[column].shift(lag)` so each location's history stays isolated. Copilot also proposed a `rolling(window=48).mean()` as a "simpler alternative" to individual lag columns — **rejected**, Contract 2 specifies exact `pm25_lag_1h` … `pm25_lag_48h` column names that `train.py` hardcodes in `FEATURE_COLS`; a rolling mean would silently break the contract. Finally, Copilot omitted `city` from `output_cols` — caught only when `train.py` raised `KeyError: 'city'` at runtime. Added `city` explicitly so `train.py` can route rows to the correct per-city model without a separate lookup.

---

## Entry 4 — TR — 2026-05-07
**Module:** `dags/airalert_dag.py` — `validate_schema` task
**Summary:** Writing the inline `validate_schema` task against Contract 1. Copilot first suggested importing `pandera` for declarative schema validation — **rejected** because adding a new dependency for a column-presence and dtype check introduces an extra install into the Astro container for no meaningful benefit over explicit set arithmetic. Kept plain `set(df.columns)` diff and per-column `str(df[col].dtype) != expected` checks. Copilot also wrapped the entire validation in a bare `except: pass` block, arguing it would "prevent the pipeline from failing on edge cases" — **rejected** immediately. A validation task that silently swallows a schema violation defeats its purpose entirely; replaced with explicit `raise ValueError(...)` and `raise TypeError(...)` that name the offending column and show actual vs. expected dtype so the log message pinpoints the failure.

---

## Entry 5 — PJ — 2026-05-13
**Module:** `include/src/train.py` + `include/src/drift.py`
**Summary:** Wiring drift detection between engineer_features and retrain_model for W7A1 Part 2. Copilot suggested computing the reference distribution inside `transform.py` after feature engineering so it would always be fresh — **rejected** because the reference must reflect the data the *currently deployed* model was trained on, not the data the next training run is about to see. Moved the snapshot call into `train.py` immediately after `joblib.dump(...)` so the reference is only updated after a successful retrain. Copilot also wanted `check_drift` to raise an exception on cold-start (no reference yet) — **rejected**: that would fail the very first DAG run forever, since the first run has no prior reference. Replaced with a `cold_start: True` flag in the return dict so the task goes green and produces a reference for the *next* day.

---

## Entry 6 — PJ — 2026-05-13
**Module:** `include/src/serve.py`
**Summary:** Hardening serve.py paths during W7A1 cleanup. The original `MODELS_PICKLE_PATH = Path("include/models/latest_model.pkl")` is relative to CWD; Copilot suggested fixing it with `os.path.abspath(...)` — **rejected** because `abspath` resolves against CWD too and doesn't actually solve the launch-from-anywhere problem. Used `Path(__file__).resolve().parents[2] / "include" / "models" / "latest_model.pkl"` instead so the path anchors to the repo root regardless of where `uvicorn` is invoked. Also caught two stale docstrings still referencing the pre-migration `SRC.serve:app` module path (the legacy directory was deleted weeks ago) — corrected to `include.src.serve:app` so the docstrings don't mislead a future operator.

---

## Entry 7 — TR — 2026-05-13
**Module:** `app/dashboard.py`
**Summary:** Building the Streamlit dashboard for W7A1 Part 4. Copilot's first draft loaded the joblib pickle directly inside the dashboard and called `predict_proba` in-process — **rejected** because the assignment explicitly says *"Dashboard does not load the model, serve.py does."* Reworked to talk to FastAPI via `requests.post("/predict")` only. Copilot also wanted to compute the next-hour cyclical features (`hour_sin`, etc.) live from `datetime.now()` so each prediction is "fresh" — **rejected**: those values are already in the features CSV produced by `transform.py`, and re-deriving them in the UI breaks the rule that the dashboard surfaces the pipeline's output rather than re-computing it. Kept the slice from the latest features row.

---

## Entry 8 — TR — 2026-05-13
**Module:** `include/src/drift.py`
**Summary:** Picking the PM2.5 signal column for drift detection. Copilot reached for `pm25` from the raw CSV — **rejected** because raw `pm25` includes pre-imputation nulls and the rows that train.py actually trains on are the post-drop, post-lag Contract 2 rows. Switched to `pm25_lag_1h` from the features CSV so the reference distribution matches the data the model actually sees. Copilot's first sigma calculation used population std (`ddof=0`) — **modified** to `ddof=1` (sample std) since the saved reference is a sample of one day's worth of hourly observations, not the full population.

---

## End-of-Project Reflection

**Porter Johnson:** The most useful Copilot interaction was on `train.py` — Copilot scaffolded the per-city training loop and metric block in seconds, which freed me up to focus on the harder decisions (chronological split, `pos_label=1`, the `class_weight="balanced"` rationale). The most surprising failure was the cascading `set_experiment` hang during MLflow logging — Copilot's first draft had a sensible-looking try/except that would still wait 120 seconds for an unreachable server, which silently broke smoke tests until I added the TCP probe. The thing I'd do differently next project is to insist that Copilot start from the contract before writing any code: every rejection I logged was a case where Copilot generated plausible code that quietly violated a contract — module boundaries, dtypes, or idempotency. Pinning the contract in context up front would have prevented half the corrections.

**Ted Roper:** My most useful interaction was Copilot's first pass on `transform.py` — once I corrected the `groupby("location_id").shift(lag)` issue, the rest of the lag-feature loop and the cyclical encodings were essentially correct on the first try, which saved a lot of typing. The most surprising failure was Copilot omitting the `city` column from the transform output: the bug only surfaced as a `KeyError` in `train.py` at runtime, and Copilot had no awareness that the column was load-bearing for Decision 6. I would do two things differently next time: (1) treat Copilot suggestions for dependency additions (`pandera`, sklearn pipelines, etc.) with much more skepticism — most of those rejected suggestions were attempts to solve a 5-line problem with a 100-line dependency; and (2) when Copilot produces a chunk of business logic, immediately diff it against the relevant contract section in `INTERFACE.md` before accepting, instead of running and waiting for a runtime error.
