"""Cost to forget one subject, under one consistent slice schedule, for the three
systems compared in the paper: from-scratch (rebuild all shards), uniform SISA
(retrain all shards), and subject-aware SISA (one shard, taken from the
per-subject speedup over all enrolled subjects).

Run:
    python experiments/08_unlearning_costs.py

Outputs:
    results/unlearning_costs.json

Reads:
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

    p_speedup = json.load(open(config.RESULTS_DIR / "unlearning_speedup.json"))
    full = p_speedup["full_build_cost"]
    sa_median_cost = full / p_speedup["speedup_median"]
    sa_mean_cost = full / p_speedup["speedup_mean"]

    # uniform: retrain all shards without one subject, same slice schedule
    ushards, _ = sisa.make_shards(sid_tr, S, R, "uniform", config.SEED)
    X0 = int(np.unique(sid_tr)[0])
    _, uinfo = unlearn.retrain_all_minus(ushards, X0, Xtr, sid_tr, Xva, sid_va,
                                         device=device, seed=config.SEED,
                                         epochs=e_i, patience=10**9)
    out = {
        "config": {"S": S, "R": R, "epochs_per_slice": e_i},
        "from_scratch_cost": full, "from_scratch_shards": S,
        "uniform_cost": uinfo["cost_steps"], "uniform_shards": uinfo["n_shards_retrained"],
        "uniform_speedup": full / uinfo["cost_steps"],
        "subject_aware_median_cost": sa_median_cost,
        "subject_aware_mean_cost": sa_mean_cost,
        "subject_aware_speedup_median": p_speedup["speedup_median"],
        "subject_aware_speedup_mean": p_speedup["speedup_mean"],
    }
    with open(config.RESULTS_DIR / "unlearning_costs.json", "w") as f:
        json.dump(out, f, indent=2)
    print("from-scratch :", f"{full:,}", f"({S} shards)")
    print("uniform      :", f"{uinfo['cost_steps']:,}", f"({uinfo['n_shards_retrained']} shards)",
          f"speedup {out['uniform_speedup']:.2f}x")
    print("subject-aware:", f"{sa_median_cost:,.0f}", "(1 shard, median)",
          f"speedup {p_speedup['speedup_median']:.1f}x")
    print("-> results/unlearning_costs.json")


if __name__ == "__main__":
    main()
