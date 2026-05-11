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

*(Add more entries as needed)*

---

## End-of-Project Reflection

*Complete before submitting your final PR — one paragraph per partner.*

**Porter Johnson:** Most useful interaction, most surprising failure, one thing you'd do differently.

**Ted Roper:** Most useful interaction, most surprising failure, one thing you'd do differently.
