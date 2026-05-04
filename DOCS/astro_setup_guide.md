# Astro CLI Setup Guide — AirAlert Project

This guide walks you through setting up your local Airflow environment using the Astronomer Astro CLI. You will need this running before Week 6 Day 2.

---

## Prerequisites

Before starting, confirm the following are already working on your machine:

- [ ] Docker Desktop is running (whale icon in taskbar, not animating)
- [ ] WSL2 active — Windows only (`wsl --list --verbose` shows an entry at VERSION 2)
- [ ] Python 3.11 virtual environment for `aml-airalert` activated in VS Code
- [ ] Your `aml-airalert` repo cloned and open in VS Code

---

## Step 1 — Install the Astro CLI

**Windows (PowerShell as Administrator):**
```powershell
winget install -e --id Astronomer.Astro
```

**Mac:**
```bash
brew install astro
```

**Verify the installation:**
```bash
astro version
```
You should see a version number (2.x or higher). If the command is not found, restart your terminal and try again.

---

## Step 2 — Initialize an Astronomer Project

Navigate to your project repo and run:

```bash
cd aml-airalert
astro dev init
```

This creates the following files and folders:

```
aml-airalert/
├── dags/               ← your DAG files go here
├── plugins/            ← leave empty
├── include/            ← leave empty
├── Dockerfile          ← do not modify
├── packages.txt        ← OS-level dependencies (leave empty)
├── requirements.txt    ← Python dependencies for Airflow
└── .astro/             ← Astro project config (gitignore this)
```

**Important:** Add the following to your `.gitignore` immediately:
```
.astro/
airflow_settings.yaml
```

Keep `Dockerfile` and `requirements.txt` committed — your team needs them.

---

## Step 3 — Add Project Dependencies

Open `requirements.txt` (the one created by `astro dev init` — this is separate from your project's root `requirements.txt`) and add:

```
apache-airflow-providers-http
requests
pandas
scikit-learn
mlflow
fastapi
uvicorn
pyarrow
```

---

## Step 4 — Start the Local Airflow Environment

```bash
astro dev start
```

This pulls Docker images and starts three containers:
- **Webserver** — the Airflow UI at `http://localhost:8080`
- **Scheduler** — watches your `dags/` folder for new DAG files
- **Database** — stores run history, XComs, and task state

**First start takes 2–4 minutes.** Subsequent starts are faster.

**Default login credentials:**
- Username: `admin`
- Password: `admin`

---

## Step 5 — Verify the Setup

1. Open `http://localhost:8080` in your browser
2. Log in with `admin` / `admin`
3. You should see the Airflow UI with an empty DAGs list

**Test that DAG detection works:**

Create a test file at `dags/hello_test.py`:
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def say_hello(**context):
    print(f"Hello from {context['ds']}")

with DAG(
    dag_id="hello_test",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    hello = PythonOperator(
        task_id="say_hello",
        python_callable=say_hello,
    )
```

Wait 30 seconds and refresh the UI. The `hello_test` DAG should appear. If it does, your environment is working correctly. Delete `dags/hello_test.py` when done.

---

## Step 6 — Stop the Environment

When you are done working:
```bash
astro dev stop
```

To restart:
```bash
astro dev start
```

---

## Common Issues

| Issue | Fix |
|---|---|
| `astro: command not found` | Restart terminal after installation |
| Docker not running error | Start Docker Desktop first, wait for it to fully load |
| Port 8080 already in use | Change the port: `astro dev start --port 8081` |
| DAG not appearing in UI | Check for Python syntax errors: `python dags/your_dag.py` |
| Import errors in task logs | Add `import sys; sys.path.insert(0, '/path/to/aml-airalert')` at the top of your task function |
| `astro dev start` hangs | Run `astro dev kill` then `astro dev start` again |
| WSL2 issues on Windows | Run `wsl --update` in PowerShell as admin, restart Docker Desktop |

---

## Important Airflow Concepts to Know Before the Lab

**`catchup=False`** — always set this. Without it, Airflow tries to run your DAG once for every day since `start_date`, which will flood your environment with hundreds of runs.

**The `dags/` folder is live** — any `.py` file you save there is picked up within 30 seconds. You do not need to restart the environment to see your changes.

**Task logs** — click any task square in the Grid view to see its log output. This is your primary debugging tool.

**Manual trigger** — click the play button (▷) next to a DAG to trigger it manually without waiting for the schedule.
