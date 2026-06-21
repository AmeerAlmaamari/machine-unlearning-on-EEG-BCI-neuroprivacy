"""Ablations under the cross-task open-set protocol: slicing R-sweep, feature
(band) ablation, and the weak-learner curve.

Run:
    python experiments/03_ablations.py

Outputs:
    results/ablations.json
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


def openset_eer_sisa(consts, mode, Xte, sid_te, Xop, classes, device):
    Mt = sisa.score_matrix(consts, mode, Xte, classes, device)
    Mo = sisa.score_matrix(consts, mode, Xop, classes, device)
    eers = []
    for j, s in enumerate(classes):
        gen = Mt[sid_te == s, j]
        if len(gen) == 0:
            continue
        eers.append(metrics.compute_eer(gen, np.concatenate([Mt[sid_te != s, j], Mo[:, j]]))[0])
    return float(np.mean(eers))


def mono_openset_eer(Xtr, ytr, Xva, yva, classes, Xte, sid_te, Xop, device, seed):
    net = model.train_mlp(Xtr, ytr, Xva, yva, len(classes), device=device, seed=seed)
    p_te = model.predict_proba(net, Xte, device=device)
    p_op = model.predict_proba(net, Xop, device=device)
    eers = []
    for j, s in enumerate(classes):
        gen = p_te[sid_te == s, j]
        if len(gen) == 0:
            continue
        eers.append(metrics.compute_eer(gen, np.concatenate([p_te[sid_te != s, j], p_op[:, j]]))[0])
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
    Xtr_f, Xva_f, Xte_f, Xop_f = (scaler.transform(X[m]).astype(np.float32) for m in (tr, va, te, op))
    sid_tr, sid_va, sid_te = sid[tr], sid[va], sid[te]
    classes = np.unique(sid_tr)
    ytr = model.encode_labels(sid_tr, classes); yva = model.encode_labels(sid_va, classes)

    # Slicing R-sweep (epochs 2e/(R+1) per slice), subject-aware S=10
    print("\n[R-sweep] slicing speedup and EER...")
    r_sweep = []
    for R in (1, 5, 10):
        e_i = max(1, round(config.MLP_EPOCHS * 2 / (R + 1)))
        shards, s2 = sisa.make_shards(sid_tr, 10, R, "subject_aware", config.SEED)
        before, info = unlearn.build_constituents(shards, Xtr_f, sid_tr, Xva_f, sid_va,
                                                  device=device, seed=config.SEED,
                                                  epochs=e_i, patience=10**9)
        eer = openset_eer_sisa(before, "subject_aware", Xte_f, sid_te, Xop_f, classes, device)
        sample = [int(s) for k in range(10) for s in shards[k]["classes"][:2]]
        costs = [unlearn.unlearn_subject_aware(shards, s2, before, Xs, Xtr_f, sid_tr,
                                               Xva_f, sid_va, device=device, seed=config.SEED,
                                               epochs=e_i, patience=10**9)["cost_steps"]
                 for Xs in sample]
        sp = info["cost_steps"] / float(np.mean(costs))
        r_sweep.append({"R": R, "open_set_eer": eer, "mean_speedup": sp})
        print(f"  R={R:2d}: open-set EER={eer:.4f} speedup={sp:.1f}x")

    # Feature (band) ablation, monolithic open-set EER
    print("\n[bands] feature ablation...")
    subsets = {"all5 (320)": list(range(config.N_FEATURES)),
               "no_gamma (256)": [c * 5 + b for c in range(64) for b in (0, 1, 2, 3)],
               "alpha+beta (128)": [c * 5 + b for c in range(64) for b in (2, 3)],
               "alpha (64)": [c * 5 + 2 for c in range(64)]}
    feat = []
    for name, cols in subsets.items():
        sc = StandardScaler().fit(X[tr][:, cols])
        Xt, Xv, Xe, Xo = (sc.transform(X[m][:, cols]).astype(np.float32) for m in (tr, va, te, op))
        eer = mono_openset_eer(Xt, ytr, Xv, yva, classes, Xe, sid_te, Xo, device, config.SEED)
        feat.append({"features": name, "open_set_eer": eer})
        print(f"  {name:18s}: open-set EER={eer:.4f}")

    # Weak-learner curve (training windows per subject)
    print("\n[budget] weak-learner curve...")
    rng = np.random.default_rng(config.SEED)
    probe = []
    for nb in (25, 50, 100, 200, None):
        if nb is None:
            keep = np.arange(len(sid_tr))
        else:
            keep = np.sort(np.concatenate([
                (lambda idx: rng.choice(idx, nb, replace=False) if len(idx) > nb else idx)(
                    np.where(sid_tr == s)[0]) for s in classes]))
        eer = mono_openset_eer(Xtr_f[keep], ytr[keep], Xva_f, yva, classes, Xte_f, sid_te,
                               Xop_f, device, config.SEED)
        lbl = nb if nb is not None else int(np.median(np.bincount(sid_tr)[classes]))
        probe.append({"windows_per_subject": int(lbl), "open_set_eer": eer})
        print(f"  budget {str(nb):>4}: open-set EER={eer:.4f}")

    out = {"protocol": "cross-task + open-set", "R_sweep": r_sweep,
           "feature_ablation": feat, "weak_learner_probe": probe}
    with open(config.RESULTS_DIR / "ablations.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n-> results/ablations.json")


if __name__ == "__main__":
    main()
