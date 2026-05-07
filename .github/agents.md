# AirAlert — Copilot Agent Instructions

> **How to use this file:** This file lives at `.github/agents.md`.
> It tells Copilot Agent what it is and is not allowed to do autonomously
> in this repository.

---

## What Copilot Agent Can Do

- Implement a **single function** that has a complete docstring and type hints
- Write **unit tests** for a function that already exists and is working
- Fix a **specific bug** when given the stack trace and the relevant function
- Refactor a function to match an **updated signature** without changing its logic
- Generate a **mock CSV** that matches a schema defined in `INTERFACE.md`
- Suggest **missing error handling** in a function that already works

---

## What Copilot Agent Must Not Do

- Generate an **entire module** from scratch — functions must be specified one at a time
- Make **schema decisions** — column names, types, and thresholds are human decisions documented in `INTERFACE.md`
- **Modify shared convention files** — see off-limits list below
- Change **column names or threshold values** without being explicitly told to
- **Skip the docstring** — every generated function must include a complete docstring before implementation
- Add **dependencies** to `requirements.txt` without being asked

---

## Files That Are Off-Limits for Autonomous Editing

Copilot Agent must never modify these files:

```
INTERFACE.md
.github/copilot-instructions.md
.github/agents.md
COPILOT_LOG.md
```

---

## Files That Require Extra Caution

Copilot Agent may suggest edits to these files but must not make structural changes without explicit instruction:

```
dags/airalert_dag.py          # task functions are OK; DAG wiring is off-limits
requirements.txt              # additions only when explicitly asked
.gitignore                    # additions only
```

---

## Output File Naming Convention

When generating any code that saves files, use this convention (W6A1: all
runtime data lives under `include/`):

```
include/data/raw/pm25_{context['ds']}.csv             # ingest.py / fetch_air_quality
include/data/features/features_{context['ds']}.csv    # transform.py / engineer_features
include/models/latest_model.pkl                       # train.py — joblib dump of {city: estimator}
include/models/metrics_{context['ds']}.json           # train.py / retrain_model — XCom return value
```

```python
# Always this pattern — not returning data directly
def my_task(**context) -> str:
    output_path = f"data/raw/pm25_{context['ds']}.csv"
    df.to_csv(output_path, index=False)
    return output_path
```

---

## XCom Rule

When generating Airflow task functions, always:

1. Save the output DataFrame to a file before returning
2. Return the file path as a string — never return a DataFrame
3. Pull input paths using `context['ti'].xcom_pull(task_ids='...')`

---

## How to Verify Generated Code

After generating any function, check:

- [ ] Column names match the schema in `copilot-instructions.md` exactly
- [ ] Return type matches the declared type hint
- [ ] Exceptions are raised (not swallowed) on failure
- [ ] If it's a DAG task, it returns a string file path
- [ ] The function does what its docstring says — run it against the mock CSV to confirm

---

## Environment Notes

- Airflow runs inside Docker via Astro CLI — do not suggest `astro dev start` or `astro dev stop` as part of code generation
- MLflow UI runs separately at `localhost:5000` — started with `mlflow ui` in a separate terminal
- FastAPI runs separately with `uvicorn serve:app --port 8000` — not part of the DAG
- All commands assume the virtual environment is activated
