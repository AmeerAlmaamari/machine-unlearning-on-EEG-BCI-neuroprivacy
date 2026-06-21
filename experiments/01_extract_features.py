"""Build the cached band-power feature table for all subjects and runs.

Run:
    python experiments/01_extract_features.py

Outputs:
    results/features.npz          (X, subject_id, run_id, window_idx, feature_names)
    results/feature_summary.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, data, features, utils  # noqa: E402


def main() -> None:
    utils.set_seed()
    subjects = data.list_subjects()
    fnames = features.feature_names()
    assert len(fnames) == config.N_FEATURES

    X_parts, sid_parts, run_parts, widx_parts = [], [], [], []
    per_subject_counts = {}

    t0 = time.time()
    for i, sid in enumerate(subjects, 1):
        n_sub = 0
        for run in data.list_runs(sid):
            X = features.extract_run(sid, run)         # [n_win, 320]
            n = X.shape[0]
            if n == 0:
                continue
            X_parts.append(X)
            sid_parts.append(np.full(n, sid, dtype=np.int16))
            run_parts.append(np.full(n, run, dtype=np.int8))
            widx_parts.append(np.arange(n, dtype=np.int16))
            n_sub += n
        per_subject_counts[sid] = n_sub
        if i % 20 == 0 or i == len(subjects):
            print(f"  {i}/{len(subjects)} subjects, "
                  f"{sum(per_subject_counts.values())} windows, {time.time()-t0:.0f}s")

    X = np.concatenate(X_parts, axis=0)
    subject_id = np.concatenate(sid_parts)
    run_id = np.concatenate(run_parts)
    window_idx = np.concatenate(widx_parts)

    np.savez_compressed(
        config.FEATURES_PATH,
        X=X, subject_id=subject_id, run_id=run_id, window_idx=window_idx,
        feature_names=np.array(fnames),
    )

    counts = np.array(list(per_subject_counts.values()))
    n_bad = int((~np.isfinite(X)).sum())
    summary = {
        "n_windows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_subjects": len(subjects),
        "windows_per_subject": {"min": int(counts.min()), "max": int(counts.max()),
                                 "mean": float(counts.mean())},
        "windows_per_run_example": int(per_subject_counts[subjects[0]] // config.N_RUNS),
        "nonfinite_values": n_bad,
        "feature_value_stats": {"min": float(X.min()), "max": float(X.max()),
                                 "mean": float(X.mean())},
        "params": {
            "window_sec": config.WINDOW_SEC, "window_samples": config.WINDOW_SAMPLES,
            "window_step": config.WINDOW_STEP, "overlap": config.WINDOW_OVERLAP,
            "bandpass": config.BANDPASS, "bands": list(config.BANDS.keys()),
            "welch_nperseg": config.WELCH_NPERSEG, "window_source": config.WINDOW_SOURCE,
        },
    }
    out = config.RESULTS_DIR / "feature_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"feature table: {X.shape[0]} windows x {X.shape[1]} features")
    print(f"windows/subject: min={counts.min()}, max={counts.max()}, mean={counts.mean():.0f}")
    print(f"non-finite values: {n_bad}")
    print(f"features -> {config.FEATURES_PATH}  ({config.FEATURES_PATH.stat().st_size/1e6:.1f} MB)")
    print(f"summary  -> {out}")


if __name__ == "__main__":
    main()
