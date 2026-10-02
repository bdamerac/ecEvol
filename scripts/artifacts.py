"""
Saving of run artifacts.

Helpers to write (a) per-sample prediction dumps as parquet, (b) small metric
tables as CSV, and (c) a run_meta.json capturing config, seed, split file, and
git commit for each run.
"""

import json
import os
import subprocess


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_preds(df, path):
    """Per-sample predictions -> parquet."""
    _ensure_parent(path)
    df.to_parquet(path, index=False)
    return path


def save_table(df, path):
    """Small summary/metric table -> CSV."""
    _ensure_parent(path)
    df.to_csv(path, index=False)
    return path


def save_run_meta(run_dir, **meta):
    """Write run_meta.json with provenance (git hash auto-added)."""
    os.makedirs(run_dir, exist_ok=True)
    payload = {**meta, "git": git_hash()}
    path = os.path.join(run_dir, "run_meta.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path
