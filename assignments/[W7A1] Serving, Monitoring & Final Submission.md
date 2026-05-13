## Overview
This assignment completes AirAlert. You will add drift detection to the pipeline, expose the model as a REST API endpoint, build a user-facing dashboard, and verify that the full system works end-to-end. You will also finalize your two remaining design decisions and submit a final pull request that documents the complete project.

By the end of this assignment, AirAlert is a deployed, monitored, and usable system — not just a model. A non-technical user should be able to open the dashboard, request a prediction, and understand the result without knowing anything about machine learning.

## Instructions
## Part 1 — Complete Decisions 7 and 8
Before implementing serve.py and dashboard.py, finalize the two deferred design decisions in INTERFACE.md.

Decision 7 (classifier and probability calibration) — you have now studied _retrain_task and understand how the model is trained. Decide what classifier you are using and how serve.py will return a meaningful unsafe_probability. If your chosen model produces uncalibrated probabilities, document how you are handling that and what the number means to a user.

Decision 8 (dashboard data sourcing) — decide how the dashboard will source its input values. Your answer should reflect what a real user of this tool would be able to do — not just what is easiest to implement.

Update the Change Log in INTERFACE.md for both decisions.

## Part 2 — Drift Detection
Add a drift check task to the Airflow DAG between _engineer_task and _retrain_task. The task should compare a recent window of PM2.5 values against a reference distribution from training and log the result to MLflow.

At minimum, track mean shift — how many standard deviations has the recent mean moved from the training mean? Log mean_shift_sigma and a boolean drifted flag as MLflow metrics. Document your drift threshold in INTERFACE.md under Decision 3's reasoning — your retraining trigger and drift threshold should be consistent with each other.

## Part 3 — FastAPI Serving Endpoint
Implement serve.py — a FastAPI application that loads the Production model from the MLflow Model Registry at startup and exposes two endpoints:

GET /health — returns the model name, stage, and a status indicator
POST /predict — accepts the feature columns from Contract 3, returns is_unsafe, unsafe_probability, and threshold_used
Your Pydantic request schema must match Contract 3 exactly. The response should reflect your Decision 7 answer — if you decided that uncalibrated probabilities are acceptable, document that in a comment in the code. If you added calibration, document how.

Run with: uvicorn serve:app --reload --port 8000

## Part 4 — Streamlit Dashboard
Implement app/dashboard.py — a Streamlit application that calls the FastAPI endpoint and presents predictions to a non-technical user.

The dashboard must include:

A health check that confirms the API is reachable before attempting a prediction
Input controls reflecting your Decision 8 choice — how the user provides feature values
A clear, plain-language display of the prediction result and confidence
A trend chart showing recent PM2.5 values with the unsafe threshold marked
The plain-language result is not optional. Displaying is_unsafe: 1 is not sufficient. A user who knows nothing about PM2.5 thresholds or machine learning should understand what the prediction means and what they should do about it.

Run with: streamlit run app/dashboard.py

Dashboard does not load the model, serve.py does.

## Part 5 — End-to-End Verification
With all services running, verify the full system works in sequence:

Trigger airalert_pipeline in the Airflow UI — all 5 tasks green including drift check
Confirm a new MLflow run is logged under the AirAlert experiment
Confirm the Production model in MLflow Registry is accessible
Make a POST request to /predict via the FastAPI Swagger UI — confirm a valid response
Open the Streamlit dashboard — make a prediction and confirm the result displays correctly
If any step fails, the log tells you where. Debug from the failure point, not from the beginning.

Also compute the naive baseline: what F1 score would a model that always predicts "safe" achieve on your validation data? Your model must beat this to demonstrate it adds value. Include this comparison in your final PR description.

## Part 6 — Final Pull Request
Open a final pull request from your working branches to main. The PR description is a professional deliverable — write it as a summary you would send a team lead at the end of a sprint.

The PR description must include:

What the system does in 1–2 sentences
Architecture summary: ingest → validate → engineer → drift check → retrain → serve → dashboard
Model performance: best F1 vs. naive baseline
Drift detection: reference distribution, threshold, and trigger logic
Copilot usage: total log entries per partner, most useful interaction, most surprising failure
What you would improve with more time

## Part 7 — Complete COPILOT_LOG.md
Each partner must have ≥4 entries total across Assignments 6 and 7. Add the end-of-project reflection section — one paragraph per partner covering most useful interaction, most surprising failure, and one thing you would do differently with Copilot on the next project.

## Learning Outcomes
Build and evaluate end-to-end machine learning pipelines — by completing the full system from automated ingestion through a deployed, user-facing prediction endpoint
Design and automate production ML workflows — by integrating drift detection into the Airflow pipeline and connecting automated retraining to a monitored serving layer
Communicate model results, technical decisions, and system limitations in plain language — by building a dashboard that presents predictions and confidence in terms a non-technical user can act on
Assess the operational risks of deployed ML systems — by implementing drift monitoring, establishing a retraining trigger, and validating the model against a naive baseline

## Deliverables
Submit the link to your shared aml-airalert GitHub repository. At the time of submission:

1. INTERFACE.md — Decisions 7 and 8 completed with reasoning; Change Log updated
2. dags/airalert_dag.py — drift check task added; all 5 tasks wire correctly
3. src/serve.py — FastAPI app running with /health and /predict functional
4. app/dashboard.py — Streamlit dashboard deployed, connected to FastAPI, trend chart present
5. COPILOT_LOG.md — ≥4 entries per partner; end-of-project reflection complete
6. Final GitHub PR open with complete description as specified above
7. All feature branches merged to main via reviewed PRs before the presentation


## In Class Notes

# Would should serve.py load the model?:

Cache with mtime check: Picks up the new model on the first request after retrain, then chaces it.

You could potentially point mtime.os.oath.getmtime(), check the timestamp before every prediction, reload the model only if it has changed since last time. You could always use the get_model function in theory, but using getmtime as well to not have to cause uncecessary bloat.

# What does the dashboard show your stakeholder?
Who is the stakeholder?
What decision does your stakeholder make?
What context makes the prediction meaningful?
What does uncertainty look like?
This is a design decision, not a techinial one. The answer should come from your stakeholder framing - not from what's easiest to display.