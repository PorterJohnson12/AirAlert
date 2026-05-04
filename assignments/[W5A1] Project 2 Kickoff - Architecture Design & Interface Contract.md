## Overview
AirAlert is not a model — it's a system. Before writing a single line of implementation code, your team needs to agree on how every part of that system will behave, what data will flow between modules, and what design choices you're making and why. This assignment is that agreement.

The decisions you make here — about data quality, feature engineering, model serving, and collaboration conventions — will shape every implementation choice that follows. A well-reasoned interface contract enables your team to build in parallel without blocking each other. A vague one leads to integration failures in Week 7 when it's hardest to fix them.

You will also configure the AI assistance layer for your project: translating your contract into context files that ensure GitHub Copilot understands your team's agreements before either partner writes a function.

## Instructions
## Part 1 — Architectural Design Decisions
Open INTERFACE.md Download INTERFACE.mdand work through the design decisions together as a pair. The decisions are split into two groups.

Complete in W5D4 (required for this assignment):

These five decisions must be answered before you write any code. They directly determine the shape of your data contracts — without them, Contract 2 cannot be completed and neither partner can build a mock CSV or start coding independently.

1. Data freshness — does this pipeline need a concept of stale data, and if so, where does it apply — training, serving, or both?
2. Missing and unreliable sensor data — how will you handle locations where sensor coverage is sparse or gapped?
3. Feature engineering choices — what lag windows, temporal features, and aggregations will transform.py produce?
4. Aggregation granularity — will the pipeline work with raw sensor readings or aggregate to one row per location per hour?
5. Single vs. multi-location model — will train.py train one global model or separate models per location?

## Defer until later weeks (flagged in INTERFACE.md, not graded in this assignment):

6. Retraining trigger (3) — what metric will you track, what threshold signals a retrain is needed, and how often?
7. Classifier choice and probability calibration — complete by W6D3 when you implement _retrain_task and begin serve.py
8. Dashboard data sourcing — complete by W7D2 when you build the dashboard

For every decision you make now: write your answer in one clear sentence, then explain your reasoning in 2–4 sentences that describe the tradeoff you considered and why your choice fits your system. There is no single right answer — the reasoning is what is being assessed.

## Part 2 — Data Contracts
With your design decisions settled, define the exact structure of data passing between each module. Complete Contracts 1 through 4 in INTERFACE.md. Every column in Contract 2 must be a deliberate decision — no blank rows.

Once Contract 2 is finalized, each partner creates a small mock CSV (5–10 rows) that matches their incoming contract boundary exactly. Save these to data/mock/ which should be gitignored. These mocks are your development foundation — you should be able to build and test your module against them before the upstream module produces real data.

## Part 3 — Context Engineering Files
Translate your finalized INTERFACE.md into the AI assistance layer:

copilot-instructions.md Download copilot-instructions.md— the project overview, stack, and Airflow conventions are pre-filled. Your job is to populate the schema section with your exact column names and types from Contracts 1 and 2, add any constants that emerged from your design decisions, and fill in your team's module ownership. This file teaches Copilot your team's agreements so every suggestion it makes is calibrated to your actual schema.

agent.md Download agent.md— standard constraints are pre-filled. Fill in your output file naming convention and any additional files you want marked as off-limits. This constrains what Copilot Agent can do autonomously in your repo.

COPILOT_LOG.md Download COPILOT_LOG.md— add the template to your repo. No entries required yet.

## Part 4 — Repository Setup and First Branch PRs
Set up the shared aml-airalert repository with the correct folder structure, .gitignore, and requirements.txt. Commit all three context files to main together as your first shared commit.

Then split to feature branches:

Person A opens a branch PR with ingest.py — at minimum a module docstring, typed function signatures, and a working OpenAQ API call
Person B opens a branch PR with transform.py — at minimum a module docstring and typed function signatures using context engineering techniques
These PRs demonstrate that both partners can work independently against the agreed contract.

## Learning Outcomes
Design and automate production ML workflows — by making explicit architectural decisions about how data flows between modules before any implementation begins
Build and evaluate end-to-end machine learning pipelines — by establishing the contract and structure that the full pipeline will be built on
Assess the operational risks of deployed ML systems — by thinking through data quality, staleness, and retraining trigger decisions that determine how the system behaves in production

## Deliverables
Submit the link to your shared aml-airalert GitHub repository. At the time of submission, the repository must contain:

1. INTERFACE.md — decisions 1–6 answered with reasoning; decisions 7–8 present but marked deferred; Contracts 1–4 complete; mock data section filled in
2. .github/copilot-instructions.md — schema section matches INTERFACE.md exactly; team names and module ownership filled in
3. .github/agents.md — output file naming convention filled in
4. COPILOT_LOG.md — template committed
5. Person A branch PR — ingest.py with module docstring, typed signatures, and working API call
6. Person B branch PR — transform.py with module docstring and typed function signatures