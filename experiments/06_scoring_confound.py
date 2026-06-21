"""Scoring control: is the subject-aware EER rise an artifact of the scorer?

The shard-local softmax score normalizes only over in-shard subjects, which could
make part of the EER rise with S an artifact of the scorer rather than an
intrinsic cost of subject-aware sharding. This script re-measures open-set EER vs
S with two scorers on the same models:
  - softmax: probability of the claimed class from the owning shard (as in paper)
  - cosine : cosine similarity between the query embedding (penultimate layer of
    the owning constituent) and the claimed subject's enrolment template (mean
    train embedding), which does not depend on in-shard normalization.
If the rise persists under cosine, the impostor-coverage cost is intrinsic.

Run:
    python experiments/06_scoring_confound.py

Outputs:
    results/scoring_confound.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, metrics, model, sisa, splits, unlearn, utils  # noqa: E402

N_HOLDOUT = 20
S_GRID = [5, 10, 20]


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

    def normalize(E):
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    rows = []
    for S in S_GRID:
        shards, s2 = sisa.make_shards(sid_tr, S, 1, "subject_aware", config.SEED)
        consts, _ = unlearn.build_constituents(shards, Xtr, sid_tr, Xva, sid_va,
                                               device=device, seed=config.SEED)
        # softmax EER (open-set)
        Mt = sisa.score_matrix(consts, "subject_aware", Xte, classes, device)
        Mo = sisa.score_matrix(consts, "subject_aware", Xop, classes, device)
        soft = []
        for j, s in enumerate(classes):
            g = Mt[sid_te == s, j]
            if len(g):
                soft.append(metrics.compute_eer(g, np.concatenate([Mt[sid_te != s, j], Mo[:, j]]))[0])
        soft_eer = float(np.mean(soft))

        # cosine EER (open-set): per shard, embed test+open once; template per subject
        cos = []
        for k, sh in enumerate(shards):
            net = consts[k]["model"]
            Ete = normalize(model.embed(net, Xte, device=device))
            Eop = normalize(model.embed(net, Xop, device=device))
            for s in [int(x) for x in sh["classes"]]:
                tmpl = model.embed(net, Xtr[sid_tr == s], device=device).mean(0)
                tmpl = tmpl / (np.linalg.norm(tmpl) + 1e-9)
                sc_te = Ete @ tmpl
                sc_op = Eop @ tmpl
                gen = sc_te[sid_te == s]
                imp = np.concatenate([sc_te[sid_te != s], sc_op])
                cos.append(metrics.compute_eer(gen, imp)[0])
        cos_eer = float(np.mean(cos))
        rows.append({"S": S, "softmax_eer": soft_eer, "cosine_eer": cos_eer})
        print(f"  S={S:2d}: softmax EER={soft_eer:.4f} | cosine EER={cos_eer:.4f}")

    out = {"protocol": "cross-task open-set", "rows": rows}
    with open(config.RESULTS_DIR / "scoring_confound.json", "w") as f:
        json.dump(out, f, indent=2)

    soft = {r["S"]: r["softmax_eer"] for r in rows}
    cos = {r["S"]: r["cosine_eer"] for r in rows}
    print(f"EER rise S=5->20: softmax {soft[20] - soft[5]:+.4f}, cosine {cos[20] - cos[5]:+.4f}")
    print("-> results/scoring_confound.json")


if __name__ == "__main__":
    main()
