"""Unlearning speedup at the headline configuration (subject-aware S=10, R=5).

Uses the SISA slicing schedule (epochs 2e/(R+1), constant total training cost
across R) and unlearns every enrolled subject, so the mean and spread of the
speedup over a full retrain are exact rather than sampled.

Run:
    python experiments/05_unlearning_speedup.py

Outputs:
    results/unlearning_speedup.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, sisa, splits, unlearn, utils  # noqa: E402

S, R = 10, 5
N_HOLDOUT = 20


def main() -> None:
    device = utils.get_device(); utils.set_seed()
    print("device:", device)
    d = np.load(config.FEATURES_PATH)
    X, sid, rid = d["X"], d["subject_id"].astype(int), d["run_id"]
    enrolled, _ = splits.holdout_subjects(N_HOLDOUT, config.SEED)
    EXEC = splits.EXECUTION_RUNS
    tr = np.isin(sid, enrolled) & np.isin(rid, EXEC[:5])
    va = np.isin(sid, enrolled) & np.isin(rid, EXEC[5:])
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xva = scaler.transform(X[tr]).astype(np.float32), scaler.transform(X[va]).astype(np.float32)
    sid_tr, sid_va = sid[tr], sid[va]

    e_i = max(1, round(config.MLP_EPOCHS * 2 / (R + 1)))
    shards, s2 = sisa.make_shards(sid_tr, S, R, "subject_aware", config.SEED)
    before, binfo = unlearn.build_constituents(shards, Xtr, sid_tr, Xva, sid_va,
                                               device=device, seed=config.SEED,
                                               epochs=e_i, patience=10**9)
    full = binfo["cost_steps"]
    print(f"epochs/slice = {e_i} | full build cost = {full:,}")

    per = []
    for k, sh in enumerate(shards):
        for Xs in [int(s) for s in sh["classes"]]:
            res = unlearn.unlearn_subject_aware(shards, s2, before, Xs, Xtr, sid_tr,
                                                Xva, sid_va, device=device, seed=config.SEED,
                                                epochs=e_i, patience=10**9)
            per.append({"subject": Xs, "shard": k, "slice": res["slice"],
                        "cost_steps": res["cost_steps"], "speedup": full / max(res["cost_steps"], 1)})

    sp = np.array([p["speedup"] for p in per])
    by_slice = {}
    for p in per:
        by_slice.setdefault(p["slice"], []).append(p["speedup"])
    out = {"config": {"S": S, "R": R, "epochs_per_slice": e_i},
           "full_build_cost": full,
           "n_subjects_unlearned": len(per),
           "speedup_mean": float(sp.mean()), "speedup_std": float(sp.std()),
           "speedup_min": float(sp.min()), "speedup_max": float(sp.max()),
           "speedup_median": float(np.median(sp)),
           "theoretical_sharding_bound": float(S),
           "speedup_by_slice": {str(k): float(np.mean(v)) for k, v in sorted(by_slice.items())}}
    with open(config.RESULTS_DIR / "unlearning_speedup.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"unlearned all {len(per)} enrolled subjects")
    print(f"speedup mean {sp.mean():.1f}x  median {np.median(sp):.1f}x  "
          f"range [{sp.min():.1f}, {sp.max():.1f}]")
    print("-> results/unlearning_speedup.json")


if __name__ == "__main__":
    main()
