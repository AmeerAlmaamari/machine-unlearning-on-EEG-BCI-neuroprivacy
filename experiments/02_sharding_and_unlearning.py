"""Sharding utility cost (S-sweep) and unlearning cost / exactness, under the
cross-task enrol-to-imagery protocol with open-set impostors.

Measures open-set EER as the number of shards grows for subject-aware and uniform
sharding, then the unlearning speedup and parameter-level exactness of removing a
subject from a subject-aware system at S=10, R=5.

Run:
    python experiments/02_sharding_and_unlearning.py

Outputs:
    results/sharding_unlearning.json
    figures/sharding_eer.png
    figures/unlearning_cost.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, metrics, model, sisa, splits, unlearn, utils  # noqa: E402

N_HOLDOUT = 20
S_GRID = [1, 5, 10, 20]
RQ2_S, RQ2_R = 10, 5


def openset_eer(consts, mode, Xte, sid_te, Xop, classes, device):
    """Mean EER with impostors = in-gallery enrolled + open-set (never-enrolled)."""
    Mt = sisa.score_matrix(consts, mode, Xte, classes, device)
    Mo = sisa.score_matrix(consts, mode, Xop, classes, device)
    eers = []
    for j, s in enumerate(classes):
        gen = Mt[sid_te == s, j]
        if len(gen) == 0:
            continue
        imp = np.concatenate([Mt[sid_te != s, j], Mo[:, j]])
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
    print(f"enrolled={len(enrolled)} held-out={len(held)} | "
          f"train={tr.sum()} test={te.sum()} open={op.sum()}")

    # S-sweep: open-set EER for both sharding modes at R=1
    print("\n[S-sweep] open-set EER by shard count...")
    s_sweep = []
    for mode in ("subject_aware", "uniform"):
        for S in S_GRID:
            shards, _ = sisa.make_shards(sid_tr, S, 1, mode, config.SEED)
            consts, _ = unlearn.build_constituents(shards, Xtr, sid_tr, Xva, sid_va,
                                                   device=device, seed=config.SEED)
            eer = openset_eer(consts, mode, Xte, sid_te, Xop, classes, device)
            s_sweep.append({"mode": mode, "S": S, "open_set_eer": eer})
            print(f"  {mode:14s} S={S:2d}: open-set EER={eer:.4f}")

    # Unlearning cost / speedup / exactness, subject-aware S=10, R=5
    print("\n[unlearning] subject-aware S=10, R=5...")
    shards, s2 = sisa.make_shards(sid_tr, RQ2_S, RQ2_R, "subject_aware", config.SEED)
    before, binfo = unlearn.build_constituents(shards, Xtr, sid_tr, Xva, sid_va,
                                               device=device, seed=config.SEED)
    targets = [int(shards[k]["classes"][0]) for k in (0, 3, 6, 9, 1)]
    per_t = []
    for Xs in targets:
        res = unlearn.unlearn_subject_aware(shards, s2, before, Xs, Xtr, sid_tr,
                                            Xva, sid_va, device=device, seed=config.SEED)
        k = res["shard"]
        fs = unlearn.train_shard_sliced(shards[k], Xtr, sid_tr, Xva, sid_va, exclude=Xs,
                                        start_slice=0, init_state=None, device=device,
                                        seed=config.SEED)
        p_u = model.predict_proba(res["model"], Xte, device=device)
        p_f = model.predict_proba(fs["model"], Xte, device=device)
        per_t.append({"subject": Xs, "shard": k, "slice": res["slice"],
                      "cost_steps": res["cost_steps"], "n_shards_retrained": 1,
                      "speedup_vs_fromscratch": binfo["cost_steps"] / max(res["cost_steps"], 1),
                      "exactness_prob_absdiff": float(np.mean(np.abs(p_u - p_f)))})
        print(f"  forget subj {Xs:3d} (shard {k}, slice {res['slice']}): "
              f"cost={res['cost_steps']:,} speedup={per_t[-1]['speedup_vs_fromscratch']:.1f}x "
              f"exact|d|={per_t[-1]['exactness_prob_absdiff']:.5f}")

    ushards, _ = sisa.make_shards(sid_tr, RQ2_S, RQ2_R, "uniform", config.SEED)
    _, uinfo = unlearn.retrain_all_minus(ushards, targets[0], Xtr, sid_tr, Xva, sid_va,
                                         device=device, seed=config.SEED)
    mean_speedup = float(np.mean([t["speedup_vs_fromscratch"] for t in per_t]))

    out = {
        "protocol": "cross-task + open-set", "n_enrolled": len(enrolled),
        "n_heldout": len(held), "S_sweep": s_sweep,
        "rq2": {"config": {"S": RQ2_S, "R": RQ2_R},
                "full_build_cost": binfo["cost_steps"],
                "subject_aware_targets": per_t, "mean_speedup": mean_speedup,
                "uniform": {"cost_steps": uinfo["cost_steps"],
                            "n_shards_retrained": uinfo["n_shards_retrained"]}},
    }
    with open(config.RESULTS_DIR / "sharding_unlearning.json", "w") as f:
        json.dump(out, f, indent=2)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for mode, col in (("subject_aware", "#4c72b0"), ("uniform", "#dd8452")):
        pts = sorted((r["S"], r["open_set_eer"]) for r in s_sweep if r["mode"] == mode)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", color=col, label=mode.replace("_", "-"))
    ax.set_xlabel("number of shards $S$"); ax.set_ylabel("open-set EER")
    ax.set_title("Sharding cost: cross-task + open-set"); ax.legend()
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "sharding_eer.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sa = float(np.mean([t["cost_steps"] for t in per_t]))
    ax.bar(["from-scratch", "uniform\nSISA", "subject-aware\nSISA"],
           [binfo["cost_steps"], uinfo["cost_steps"], sa],
           color=["#7f7f7f", "#dd8452", "#4c72b0"])
    ax.set_yscale("log"); ax.set_ylabel("retraining cost (epochs x samples)")
    ax.set_title("Cost to forget one subject")
    for i, (v, s) in enumerate([(binfo["cost_steps"], RQ2_S), (uinfo["cost_steps"], RQ2_S), (sa, 1)]):
        ax.text(i, v, f"{s} shard(s)", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "unlearning_cost.png", dpi=120); plt.close(fig)

    print(f"\nmean speedup={mean_speedup:.1f}x | uniform shards retrained={uinfo['n_shards_retrained']}")
    print("-> results/sharding_unlearning.json + figures/sharding_eer.png, unlearning_cost.png")


if __name__ == "__main__":
    main()
