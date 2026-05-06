## Overview
This assignment is where AirAlert becomes a real automated system. You will implement the four Airflow DAG tasks that form the core of your pipeline, wire them together, and verify that the full pipeline runs end-to-end without errors. You will also complete your final outstanding design decision and begin documenting your AI assistance interactions.

The pipeline is only as strong as its weakest module boundary. A task that silently passes the wrong data to the next task is harder to debug than one that fails loudly. Your goal is not just a green pipeline — it is a pipeline you can explain and defend.

## Instructions
## Part 1 — Complete Decision 3
Finalize the retraining trigger decision in INTERFACE.md before writing any train.py logic. Your answer must address: what metric you track, what threshold triggers a retrain, and why that threshold reflects the cost of a false negative in a public health context.

Use F1 on the unsafe class — not accuracy. A model that always predicts "safe" can achieve high accuracy while being completely useless. Update the Change Log if this revises anything from Assignment 5.

## Part 2 — Implement the DAG tasks and pipeline scripts
Study retrain_model in the skeleton before writing your own tasks. It shows the correct pattern.

Every task must:

Use the @task decorator — no PythonOperator, no manual xcom_push / xcom_pull
Return a string file path — never a DataFrame or None
Get the execution date via get_current_context()["ds"] — not datetime.now()
Check idempotency — return early if the output file already exists
Raise a meaningful exception on failure — except: pass is not acceptable
Task and script ownership
Decide between yourselves who owns which tasks and scripts before writing any code. Document your split in INTERFACE.md under Module Ownership — both partners must sign off on it before work begins.

The only constraints:

Each partner owns at least one DAG task and its corresponding pipeline script
Ownership must be roughly balanced — one partner cannot own a single trivial task while the other owns three complex ones
The reviewer of a task must not be the same person who wrote it
Task and script reference

## DAG task	Pipeline script	What it does
fetch_air_quality	include/src/ingest.py	Calls OpenAQ and Open-Meteo APIs, merges on (location_id, timestamp), saves raw CSV. Falls back to synthetic data if OpenAQ is unavailable — your task does not need to handle this separately.
validate_schema	(inline in DAG or shared utility)	Checks every Contract 1 column is present with the correct type. Pass-through on success — fail loudly on any violation.
engineer_features	include/src/transform.py	Creates feature columns and target variable per Contract 2. Output column names and types must match exactly.
retrain_model	include/src/train.py	Provided complete — study it, do not modify it.
Part 3 — Cross-review Both
Review your partner's tasks as inline comments on their GitHub PR. Each item below must be checked — leave a comment for anything that fails. The PR owner addresses all comments before merging.

If you use the GitHub Copilot automated review, annotate the Copilot summary with agreed / disagreed / N/A and one sentence of reasoning per item. An unannotated Copilot summary does not count as a review.

Returns a string file path — not a DataFrame or None
Output file saved before the return statement
Filename includes execution date from get_current_context(), not datetime.now()
Idempotency check present
Column names exactly match the relevant contract in INTERFACE.md
Lag features grouped by location_id
Meaningful exception raised on failure
Paths use include/data/ — not data/ at the repo root
No hardcoded API keys or absolute local paths
Part 4 — Verify the full pipeline Both
Restore full task wiring and trigger airalert_pipeline manually in the Airflow UI.

python
raw = fetch_air_quality()
validated = validate_schema(raw)
features = engineer_features(validated)
retrain_model(features)
All four tasks must go green in sequence. Screenshot the Grid view and save as docs/airflow_green.png. If a task fails, read the log — the error message tells you exactly what broke.

After a green run, confirm:

XCom on retrain_model — return_value contains f1, baseline_f1, accuracy, precision, recall
include/data/raw/pm25_{date}.csv — exists, 9 columns, multiple cities
include/data/features/features_{date}.csv — exists, lag columns present, is_unsafe column present
include/models/latest_model.pkl — exists and loads with joblib.load()
Part 5 — COPILOT_LOG.md Both — ≥2 entries each
Document significant Copilot interactions from this week. Each entry must include what you accepted, and — more importantly — what you rejected or modified and why. The most valuable entry is one where you disagreed with Copilot and can explain why it was wrong.

For GitHub Copilot PR reviews, your log entry is your per-item annotation of the review summary.

## Part 6 — Lab pass-off Both — in person
During W6D3 or W6D4, your instructor will ask each partner:

## IMPORTANT: WRITTEN OUT IN AN MD FILE AND GIVEN SUGGESTIONS BEFORE IN PERSON REVIEW
Walk me through one task you implemented — what does it do, what does it return, and what happens if the upstream data is missing?
Show me one place where your implementation reflects a decision from INTERFACE.md.
If you used the GitHub Copilot PR review, you may also be asked: "You left a comment on line X — walk me through why that was a problem."

## Deliverables
File	Requirement
INTERFACE.md	Decision 3 complete with reasoning; Change Log updated
dags/airalert_dag.py	All 4 tasks, full wiring, @task decorator throughout
include/src/ingest.py, transform.py	Scripts in include/src/, paths use include/data/
docs/airflow_green.png	All 4 tasks green in the Grid view
COPILOT_LOG.md	≥2 entries per partner; Copilot PR review entries annotated
Pull requests	Both PRs show cross-review comments addressed before merge
Lab pass-off	Completed in person during W6 lab sessions