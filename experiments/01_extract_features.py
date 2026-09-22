from __future__ import annotations

import json
import sys
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
    for sid in subjects:
        n_sub = 0
        for run in data.list_runs(sid):
            X = features.extract_run(sid, run)
            n = X.shape[0]
            if n == 0:
                continue
            X_parts.append(X)
            sid_parts.append(np.full(n, sid, dtype=np.int16))
            run_parts.append(np.full(n, run, dtype=np.int8))
            widx_parts.append(np.arange(n, dtype=np.int16))
            n_sub += n
        per_subject_counts[sid] = n_sub

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
    summary = {
        "n_windows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_subjects": len(subjects),
        "windows_per_subject": {"min": int(counts.min()), "max": int(counts.max()),
                                "mean": float(counts.mean())},
        "nonfinite_values": int((~np.isfinite(X)).sum()),
        "feature_value_stats": {"min": float(X.min()), "max": float(X.max()),
                                "mean": float(X.mean())},
        "params": {
            "window_sec": config.WINDOW_SEC, "window_samples": config.WINDOW_SAMPLES,
            "window_step": config.WINDOW_STEP, "overlap": config.WINDOW_OVERLAP,
            "bandpass": config.BANDPASS, "bands": list(config.BANDS.keys()),
            "welch_nperseg": config.WELCH_NPERSEG, "window_source": config.WINDOW_SOURCE,
        },
    }
    with open(config.RESULTS_DIR / "feature_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
