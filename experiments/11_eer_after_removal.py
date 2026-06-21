"""Does forgetting one subject change the authentication accuracy for the subjects
who remain?

Protocol: subject-aware S=10, R=5, cross-task + open-set, seed 42. For each removed
target X:
  1. baseline = mean open-set EER over the remaining enrolled subjects (all
     enrolled except X), using the original constituents.
  2. after    = same metric, but the constituent of X's home shard is replaced by
     the exactly unlearned constituent (X removed); all other shards unchanged.
The removed subject X is excluded as a genuine target and from the impostor pool
in both measurements, so the two numbers are over the same remaining subjects and
differ only by the unlearning of shard k.

Run:
    python experiments/11_eer_after_removal.py

Outputs:
    results/eer_after_removal.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, metrics, sisa, splits, unlearn, utils  # noqa: E402

S, R = 10, 5
N_HOLDOUT = 20


def openset_eer_remaining(consts, Xte, sid_te, Xop, classes, device, exclude):
    """Mean open-set EER over enrolled subjects, excluding `exclude` as a genuine
    target and from the enrolled impostor pool. Held-out open-set impostors stay."""
    Mt = sisa.score_matrix(consts, "subject_aware", Xte, classes, device)
    Mo = sisa.score_matrix(consts, "subject_aware", Xop, classes, device)
    keep_te = sid_te != exclude
    eers = []
    for j, s in enumerate(classes):
        if int(s) == int(exclude):
            continue
        gen = Mt[(sid_te == s), j]
        if len(gen) == 0:
            continue
        imp = np.concatenate([Mt[keep_te & (sid_te != s), j], Mo[:, j]])
        eers.append(metrics.compute_eer(gen, imp)[0])
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

    shards, s2 = sisa.make_shards(sid_tr, S, R, "subject_aware", config.SEED)
    before, _ = unlearn.build_constituents(shards, Xtr, sid_tr, Xva, sid_va,
                                           device=device, seed=config.SEED)
    # one target per shard (first subject), so every shard is exercised once
    targets = [int(shards[k]["classes"][0]) for k in range(S)]

    per = []
    for X0 in targets:
        k = s2[X0]
        res = unlearn.unlearn_subject_aware(shards, s2, before, X0, Xtr, sid_tr,
                                            Xva, sid_va, device=device, seed=config.SEED)
        after = list(before)
        after[k] = {"model": res["model"], "classes": res["classes"]}
        e_base = openset_eer_remaining(before, Xte, sid_te, Xop, classes, device, X0)
        e_after = openset_eer_remaining(after, Xte, sid_te, Xop, classes, device, X0)
        per.append({"removed": X0, "shard": k, "slice": res["slice"],
                    "eer_remaining_before": e_base, "eer_remaining_after": e_after,
                    "delta": e_after - e_base})
        print(f"  remove subj {X0:3d} (shard {k}): remaining EER "
              f"{e_base:.4f} -> {e_after:.4f}  (delta {e_after - e_base:+.5f})")

    base = np.array([p["eer_remaining_before"] for p in per])
    aft = np.array([p["eer_remaining_after"] for p in per])
    out = {"config": {"S": S, "R": R}, "n_removals": len(per),
           "mean_eer_remaining_before": float(base.mean()),
           "mean_eer_remaining_after": float(aft.mean()),
           "mean_abs_delta": float(np.mean(np.abs(aft - base))),
           "max_abs_delta": float(np.max(np.abs(aft - base))),
           "per_removal": per}
    with open(config.RESULTS_DIR / "eer_after_removal.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"mean remaining EER  before={base.mean():.4f}  after={aft.mean():.4f}")
    print(f"mean |delta|={out['mean_abs_delta']:.5f}  max |delta|={out['max_abs_delta']:.5f}")
    print("-> results/eer_after_removal.json")


if __name__ == "__main__":
    main()
