"""
drift.py — AirAlert PM2.5 distribution drift detection (W7A1 Part 2).

Persists training PM2.5 statistics during ``retrain_model`` and compares each
day's incoming feature distribution against that reference so the DAG can
log a ``mean_shift_sigma`` metric and a ``drifted`` boolean to MLflow.

Reference is written by include/src/train.py after a successful model dump;
the next day's ``drift_check`` task reads it back. On the first ever run the
reference does not yet exist — ``check_drift`` returns ``cold_start=True``
and a zero shift so the task still goes green (per W6A1 task conventions).

Owner: Ted Roper
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# ── Constants (from INTERFACE.md) ─────────────────────────────────────────────

# A 2-σ shift is the standard "unusual" boundary and pairs with Decision 3's
# F1<0.60 retrain trigger: 2-σ drift is the early-warning signal that data is
# moving; F1<0.60 is the act-now signal that performance has actually fallen.
DRIFT_SIGMA_THRESHOLD: float = 2.0

# transform.py emits one row per (location, hour) — the lag-1 column is the
# most-recent pm25 reading for that row. We use it as the rolling pm25 signal
# rather than reading the raw CSV so the drift check stays bound to whatever
# data the model is actually trained on.
PM25_SIGNAL_COL: str = "pm25_lag_1h"

DEFAULT_REFERENCE_PATH: Path = Path("include/models/reference_pm25_stats.json")


# ── Core functions ────────────────────────────────────────────────────────────


def compute_pm25_stats(features_path: str | Path) -> dict[str, float]:
    """Compute mean, std, and count of the PM2.5 signal in a features CSV.

    Args:
        features_path: Path to a Contract 2 features CSV.

    Returns:
        Dict with float ``mean``, float ``std`` (sample std, ddof=1), and int ``n``.

    Raises:
        FileNotFoundError: If ``features_path`` does not exist.
        ValueError: If the CSV is empty or missing the PM25_SIGNAL_COL column.
    """
    p = Path(features_path)
    if not p.exists():
        raise FileNotFoundError(f"Features CSV not found: {p}")

    df = pd.read_csv(p)
    if PM25_SIGNAL_COL not in df.columns:
        raise ValueError(
            f"Column {PM25_SIGNAL_COL!r} missing from {p}. "
            "drift.py expects a Contract 2 features CSV."
        )
    series = df[PM25_SIGNAL_COL].dropna()
    if series.empty:
        raise ValueError(f"No usable rows in {p} for {PM25_SIGNAL_COL!r}.")

    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)),
        "n": int(len(series)),
    }


def save_reference(
    stats: dict[str, float],
    path: str | Path = DEFAULT_REFERENCE_PATH,
) -> str:
    """Persist a reference distribution snapshot for future drift checks.

    Called from include/src/train.py after a successful pickle dump so the
    reference always reflects the data the latest model was trained on.

    Args:
        stats: Output of :func:`compute_pm25_stats`.
        path:  Destination JSON path.

    Returns:
        Absolute string path to the written JSON.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    return str(out)


def check_drift(
    features_path: str | Path,
    reference_path: str | Path = DEFAULT_REFERENCE_PATH,
    sigma_threshold: float = DRIFT_SIGMA_THRESHOLD,
) -> dict[str, Any]:
    """Compare the current features distribution against the saved reference.

    ``mean_shift_sigma`` is computed as ``(recent_mean - reference_mean) /
    reference_std``. A zero reference std (degenerate single-value reference)
    is treated as a 0-shift so the task does not divide by zero. When the
    reference file is absent (first ever run, or models dir was wiped) this
    returns a cold-start result rather than raising — the task is informational
    on the first run and only earns its keep from day 2 onward.

    Args:
        features_path:    Today's features CSV (Contract 2).
        reference_path:   JSON file written by :func:`save_reference`.
        sigma_threshold:  ``drifted=True`` when ``abs(mean_shift_sigma)`` exceeds this.

    Returns:
        Dict with ``mean_shift_sigma`` (float), ``drifted`` (bool),
        ``reference_mean`` (float | None), ``recent_mean`` (float), ``n_recent``
        (int), and ``cold_start`` (bool). The MLflow-logged metrics are
        ``mean_shift_sigma`` and ``drifted`` per W7A1 Part 2.

    Raises:
        FileNotFoundError: If ``features_path`` does not exist.
        ValueError: If the features CSV is malformed (see compute_pm25_stats).
    """
    recent = compute_pm25_stats(features_path)
    ref_p = Path(reference_path)

    if not ref_p.exists():
        return {
            "mean_shift_sigma": 0.0,
            "drifted": False,
            "reference_mean": None,
            "reference_std": None,
            "recent_mean": recent["mean"],
            "recent_std": recent["std"],
            "n_recent": recent["n"],
            "cold_start": True,
        }

    reference = json.loads(ref_p.read_text())
    ref_std = float(reference.get("std", 0.0))
    ref_mean = float(reference.get("mean", 0.0))

    if ref_std == 0.0:
        mean_shift_sigma = 0.0
    else:
        mean_shift_sigma = (recent["mean"] - ref_mean) / ref_std

    return {
        "mean_shift_sigma": float(mean_shift_sigma),
        "drifted": bool(abs(mean_shift_sigma) > sigma_threshold),
        "reference_mean": ref_mean,
        "reference_std": ref_std,
        "recent_mean": recent["mean"],
        "recent_std": recent["std"],
        "n_recent": recent["n"],
        "cold_start": False,
    }


# ── Local dev entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    features = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "include/data/features/features_2026-04-25.csv"
    )
    result = check_drift(features)
    print(json.dumps(result, indent=2))
