"""Non-saturated, model-based re-identification: target-centric rank-1 over the
full 103-subject population computed on model embeddings.

Attack model: the adversary holds one constituent (the unlearned shard model) and
a labelled gallery for every candidate subject. For each candidate s the adversary
forms a template = L2-normalized mean penultimate-layer embedding of s's gallery
windows (imagery runs 2,4,6). A probe window (imagery runs 8,10,12) is assigned to
the nearest template by cosine. Rank-1 = fraction of the target's probe windows
assigned to the target, over all 103 candidate templates.

Four conditions per removed target X (home shard k):
  before : embeddings from the constituent trained with X         (positive control)
  after  : embeddings from the constituent after exact removal of X
  floor  : same after-model, but the target is a never-enrolled subject N
  raw    : same matcher on the z-scored band-power features (no model)

Difficulty is swept by channel count (64/16/8/4). Setup, shards, seed, and 16
targets (2 per shard, S=8) match the AUC audit, so the numbers are comparable.

Run:
    python experiments/10_reid_rank1.py

Outputs:
    results/reid_rank1.json
    figures/reid_rank1.png
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
from src import config, model, sisa, splits, unlearn, utils  # noqa: E402

S, R = 8, 5
N_HOLDOUT = 20
CHANNELS = [64, 16, 8, 4]
GAL_RUNS, PROBE_RUNS = [2, 4, 6], [8, 10, 12]
N_TARGETS = 16


def select_cols(c):
    idx = np.unique(np.linspace(0, 63, c).astype(int))
    cols = [ch * 5 + b for ch in idx for b in range(5)]
    return cols, len(idx)


def _norm(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


def templates(feat, sid, subjects):
    """Per-subject L2-normalized mean template over `subjects` (gallery rows)."""
    T, tids = [], []
    for s in subjects:
        g = feat[sid == s]
        if len(g):
            T.append(g.mean(0)); tids.append(int(s))
    return _norm(np.asarray(T, dtype=np.float64)), np.asarray(tids)


def rank1_target(gal_feat, gal_sid, subjects, probe_feat, target_id):
    """Fraction of the target's probe windows whose nearest gallery template (by
    cosine, over all `subjects`) is the target. Population size = #templates."""
    if len(probe_feat) == 0:
        return float("nan"), 0
    T, tids = templates(gal_feat, gal_sid, subjects)
    P = _norm(np.asarray(probe_feat, dtype=np.float64))
    pred = tids[np.argmax(P @ T.T, axis=1)]
    return float(np.mean(pred == int(target_id))), int(len(tids))


def main() -> None:
    device = utils.get_device(); utils.set_seed()
    print("device:", device)
    d = np.load(config.FEATURES_PATH)
    Xall, sid, rid = d["X"], d["subject_id"].astype(int), d["run_id"]
    enrolled, held = splits.holdout_subjects(N_HOLDOUT, config.SEED)
    EXEC, IMAG = splits.EXECUTION_RUNS, splits.IMAGERY_RUNS
    tr = np.isin(sid, enrolled) & np.isin(rid, EXEC[:5])
    va = np.isin(sid, enrolled) & np.isin(rid, EXEC[5:])
    all_subjects = np.array(sorted(set(enrolled) | set(held)))   # 103 candidates
    gal = np.isin(rid, GAL_RUNS)      # gallery rows over all subjects
    prb = np.isin(rid, PROBE_RUNS)    # probe rows over all subjects

    rows = []
    for c in CHANNELS:
        cols, nch = select_cols(c)
        Xc = Xall[:, cols]
        scaler = StandardScaler().fit(Xc[tr])
        Xtr = scaler.transform(Xc[tr]).astype(np.float32)
        Xva = scaler.transform(Xc[va]).astype(np.float32)
        Xsc = scaler.transform(Xc).astype(np.float32)   # all rows, z-scored
        sid_tr, sid_va = sid[tr], sid[va]

        # raw gallery/probe (z-scored band power, no model)
        Xgal_raw, sgal = Xsc[gal], sid[gal]
        Xprb_raw, sprb, rprb = Xsc[prb], sid[prb], rid[prb]

        shards, s2 = sisa.make_shards(sid_tr, S, R, "subject_aware", config.SEED)
        before, _ = unlearn.build_constituents(shards, Xtr, sid_tr, Xva, sid_va,
                                               device=device, seed=config.SEED)
        targets = [int(shards[k]["classes"][j]) for k in range(S)
                   for j in range(N_TARGETS // S)]

        # cache per-shard "before" gallery embeddings (2 targets share a shard)
        emb = lambda net, A: model.embed(net, A, device=device)
        before_gal_emb = {}

        b = {"raw": [], "before": [], "after": [], "floor": []}
        pops = []
        for i, X in enumerate(targets):
            k = s2[X]
            res = unlearn.unlearn_subject_aware(shards, s2, before, X, Xtr, sid_tr,
                                                Xva, sid_va, device=device, seed=config.SEED)
            after_net = res["model"]
            before_net = before[k]["model"]

            # gallery/probe embeddings through each model
            if k not in before_gal_emb:
                before_gal_emb[k] = emb(before_net, Xgal_raw)
            bg = before_gal_emb[k]
            ag = emb(after_net, Xgal_raw)

            # target probe windows (X) through each model
            xpm = sprb == X
            Xprb_X = Xprb_raw[xpm]
            bx = emb(before_net, Xprb_X)
            ax = emb(after_net, Xprb_X)

            # never-enrolled floor subject N (same rotation as the AUC audit)
            N = held[i % len(held)]
            npm = sprb == N
            an_gal = ag                      # same after-model gallery
            an_prb = emb(after_net, Xprb_raw[npm])

            r_raw, pop = rank1_target(Xgal_raw, sgal, all_subjects, Xprb_X, X)
            r_bef, _ = rank1_target(bg, sgal, all_subjects, bx, X)
            r_aft, _ = rank1_target(ag, sgal, all_subjects, ax, X)
            r_flr, _ = rank1_target(an_gal, sgal, all_subjects, an_prb, N)
            b["raw"].append(r_raw); b["before"].append(r_bef)
            b["after"].append(r_aft); b["floor"].append(r_flr)
            pops.append(pop)

        def ms(key):
            a = np.array(b[key], dtype=float)
            a = a[np.isfinite(a)]
            return float(a.mean()), float(a.std())

        row = {"channels": nch, "n_targets": len(targets),
               "population": int(np.median(pops)), "chance": 1.0 / float(np.median(pops))}
        for key in ("raw", "before", "after", "floor"):
            m, s_ = ms(key)
            row[f"{key}_rank1"] = m
            row[f"{key}_rank1_std"] = s_
        row["after_per_subject"] = b["after"]
        row["floor_per_subject"] = b["floor"]
        rows.append(row)
        print(f"  c={nch:2d}ch (pop {row['population']}, chance {row['chance']:.3f}): "
              f"raw={row['raw_rank1']:.3f} before={row['before_rank1']:.3f} "
              f"after={row['after_rank1']:.3f} floor={row['floor_rank1']:.3f}")

    out = {"protocol": "target-centric rank-1 over 103 subjects on model embeddings; "
                       "gallery imagery runs 2,4,6 / probe runs 8,10,12; "
                       "channel-reduction difficulty sweep",
           "S": S, "R": R, "n_targets": N_TARGETS, "rows": rows}
    with open(config.RESULTS_DIR / "reid_rank1.json", "w") as f:
        json.dump(out, f, indent=2)

    xs = [r["channels"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for key, lab, col in [("raw", "raw features", "#8172b3"),
                          ("before", "before", "#4c72b0"),
                          ("after", "after (unlearned)", "#c44e52"),
                          ("floor", "never-enrolled floor", "#55a868")]:
        ys = [r[f"{key}_rank1"] for r in rows]
        es = [r[f"{key}_rank1_std"] for r in rows]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=lab, color=col)
    ax.axhline(rows[0]["chance"], ls="--", c="k", lw=1, label="chance (1/103)")
    ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels(xs)
    ax.set_xlabel("number of EEG channels")
    ax.set_ylabel("rank-1 identification over 103 subjects")
    ax.set_title("Model-embedding re-identification (non-saturated)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "reid_rank1.png", dpi=120)
    plt.close(fig)
    print("-> results/reid_rank1.json + figures/reid_rank1.png")


if __name__ == "__main__":
    main()
