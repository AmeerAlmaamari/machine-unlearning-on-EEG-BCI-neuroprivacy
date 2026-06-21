"""Multi-seed variance for the main-table EER numbers.

Reports the mean and standard deviation of the open-set cross-task EER over five
seeds (varying shard assignment and model initialization; the enrolled/held-out
split is fixed) for the three systems in the main table.

Run:
    python experiments/07_multiseed_eer.py

Outputs:
    results/multiseed_eer.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, metrics, sisa, splits, unlearn, utils  # noqa: E402

N_HOLDOUT = 20
SEEDS = [42, 43, 44, 45, 46]


def openset_eer(consts, mode, Xte, sid_te, Xop, classes, device):
    Mt = sisa.score_matrix(consts, mode, Xte, classes, device)
    Mo = sisa.score_matrix(consts, mode, Xop, classes, device)
    eers = []
    for j, s in enumerate(classes):
        g = Mt[sid_te == s, j]
        if len(g):
            eers.append(metrics.compute_eer(g, np.concatenate([Mt[sid_te != s, j], Mo[:, j]]))[0])
    return float(np.mean(eers))


def main() -> None:
    device = utils.get_device(); utils.set_seed()
    print("device:", device)
    d = np.load(config.FEATURES_PATH)
    X, sid, rid = d["X"], d["subject_id"].astype(int), d["run_id"]
    enrolled, held = splits.holdout_subjects(N_HOLDOUT, config.SEED)
    EXEC, IMAG = splits.EXECUTION_RUNS, splits.IMAGERY_RUNS
    tr = np.isin(sid, enrolled) & np.isin(rid, EXEC[:5])
    va = np.isin(sid, enrolled) & np.isin(rid, EXEC[5:])
    te = np.isin(sid, enrolled) & np.isin(rid, IMAG)
    op = np.isin(sid, held) & np.isin(rid, IMAG)
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xva, Xte, Xop = (scaler.transform(X[m]).astype(np.float32) for m in (tr, va, te, op))
    sid_tr, sid_va, sid_te = sid[tr], sid[va], sid[te]
    classes = np.unique(sid_tr)

    systems = [("monolithic", "subject_aware", 1),
               ("subject_aware_S10", "subject_aware", 10),
               ("uniform_S10", "uniform", 10)]
    res = {name: [] for name, _, _ in systems}
    for seed in SEEDS:
        for name, mode, Sv in systems:
            shards, _ = sisa.make_shards(sid_tr, Sv, 1, mode, seed)
            consts, _ = unlearn.build_constituents(shards, Xtr, sid_tr, Xva, sid_va,
                                                   device=device, seed=seed)
            res[name].append(openset_eer(consts, mode, Xte, sid_te, Xop, classes, device))
        print(f"  seed {seed}: " + " | ".join(f"{n}={res[n][-1]:.4f}" for n, _, _ in systems))

    out = {"seeds": SEEDS,
           "systems": {n: {"mean_eer": float(np.mean(res[n])), "std_eer": float(np.std(res[n])),
                           "per_seed": res[n]} for n, _, _ in systems}}
    with open(config.RESULTS_DIR / "multiseed_eer.json", "w") as f:
        json.dump(out, f, indent=2)

    for n, _, _ in systems:
        print(f"  {n:20s}: {np.mean(res[n]):.4f} +/- {np.std(res[n]):.4f}")
    print("-> results/multiseed_eer.json")


if __name__ == "__main__":
    main()
