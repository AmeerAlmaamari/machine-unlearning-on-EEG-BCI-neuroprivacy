"""Leakage-free re-identification audit (separability AUC and raw rank-1).

The probe is run-disjoint: it is trained on imagery runs {2,4,6} and tested on
disjoint imagery runs {8,10,12}, so overlapping windows cannot cross the split.
Difficulty is swept by channel count (64/16/8/4) via column-selecting the cached
band-power, since fewer channels lower identifiability and consumer headsets have
few channels.

Conditions per removed subject: before (model trained with X), after (exact
removal), floor (a never-enrolled subject under the same model), raw (the
band-power features, no model). A rank-1 re-identification over the full
population on raw features is also reported.

Run:
    python experiments/04_reid_auc.py

Outputs:
    results/reid_auc.json
    figures/reid_auc_channels.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model, sisa, splits, unlearn, utils  # noqa: E402

S, R = 8, 5
N_HOLDOUT = 20
CHANNELS = [64, 16, 8, 4]
GAL_RUNS, PROBE_RUNS = [2, 4, 6], [8, 10, 12]   # run-disjoint within imagery
N_TARGETS = 16
BG_CAP = 3000


def select_cols(c):
    idx = np.unique(np.linspace(0, 63, c).astype(int))
    cols = [ch * 5 + b for ch in idx for b in range(5)]
    return cols, len(idx)


def sep_rundisjoint(Xt, rt, Xb, rb, seed):
    """Run-disjoint binary separability AUC: probe trained on GAL_RUNS, tested on
    PROBE_RUNS, balanced target vs background."""
    rng = np.random.default_rng(seed)

    def part(runs):
        ti, bi = np.isin(rt, runs), np.isin(rb, runs)
        A, B = Xt[ti], Xb[bi]
        n = min(len(A), len(B))
        if n < 5:
            return None
        A = A[rng.choice(len(A), n, replace=False)]
        B = B[rng.choice(len(B), n, replace=False)]
        return np.vstack([A, B]), np.r_[np.ones(n), np.zeros(n)]

    tr, te = part(GAL_RUNS), part(PROBE_RUNS)
    if tr is None or te is None:
        return float("nan")
    sc = StandardScaler().fit(tr[0])
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(tr[0]), tr[1])
    return float(roc_auc_score(te[1], clf.predict_proba(sc.transform(te[0]))[:, 1]))


def rank1_population(feat, sid, rid, subjects, seed, n_probe=2000):
    """Rank-1 re-identification on raw features: per-subject gallery template from
    GAL_RUNS, single-window probes from PROBE_RUNS matched over the whole
    population by nearest template (cosine)."""
    rng = np.random.default_rng(seed)
    templ, tids = [], []
    for s in subjects:
        g = feat[(sid == s) & np.isin(rid, GAL_RUNS)]
        if len(g):
            templ.append(g.mean(0)); tids.append(s)
    T = np.array(templ); tids = np.array(tids)
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    pmask = np.isin(rid, PROBE_RUNS)
    P, ps = feat[pmask], sid[pmask]
    idx = rng.choice(len(P), min(n_probe, len(P)), replace=False)
    P, ps = P[idx], ps[idx]
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    correct = (tids[np.argmax(P @ T.T, axis=1)] == ps)
    acc = float(correct.mean())
    ci = 1.96 * float(np.sqrt(acc * (1 - acc) / len(correct)))  # binomial 95% CI
    return acc, len(tids), ci


def main() -> None:
    device = utils.get_device(); utils.set_seed()
    print("device:", device)
    d = np.load(config.FEATURES_PATH)
    Xall, sid, rid = d["X"], d["subject_id"].astype(int), d["run_id"]
    enrolled, held = splits.holdout_subjects(N_HOLDOUT, config.SEED)
    EXEC, IMAG = splits.EXECUTION_RUNS, splits.IMAGERY_RUNS
    tr = np.isin(sid, enrolled) & np.isin(rid, EXEC[:5])
    va = np.isin(sid, enrolled) & np.isin(rid, EXEC[5:])
    te = np.isin(sid, enrolled) & np.isin(rid, IMAG)
    op = np.isin(sid, held) & np.isin(rid, IMAG)
    all_subjects = sorted(set(enrolled) | set(held))

    rows = []
    for c in CHANNELS:
        cols, nch = select_cols(c)
        Xc = Xall[:, cols]
        scaler = StandardScaler().fit(Xc[tr])
        Xtr, Xva = scaler.transform(Xc[tr]).astype(np.float32), scaler.transform(Xc[va]).astype(np.float32)
        Xte, Xop = scaler.transform(Xc[te]).astype(np.float32), scaler.transform(Xc[op]).astype(np.float32)
        Xfull = scaler.transform(Xc).astype(np.float32)  # for rank-1 over all windows
        sid_tr, sid_va = sid[tr], sid[va]
        sid_te, rid_te = sid[te], rid[te]
        sid_op, rid_op = sid[op], rid[op]
        classes = np.unique(sid_tr)

        shards, s2 = sisa.make_shards(sid_tr, S, R, "subject_aware", config.SEED)
        before, _ = unlearn.build_constituents(shards, Xtr, sid_tr, Xva, sid_va,
                                               device=device, seed=config.SEED)
        targets = [int(shards[k]["classes"][j]) for k in range(S) for j in range(N_TARGETS // S)]

        rng = np.random.default_rng(config.SEED)
        b = {"raw": [], "before": [], "after": [], "floor": []}
        for i, Xs in enumerate(targets):
            k = s2[Xs]
            res = unlearn.unlearn_subject_aware(shards, s2, before, Xs, Xtr, sid_tr,
                                                Xva, sid_va, device=device, seed=config.SEED)
            bg_sub = np.array([s for s in enrolled if s != Xs])
            bgm = np.isin(sid_te, bg_sub)
            Xbg, rbg = Xte[bgm], rid_te[bgm]
            if len(Xbg) > BG_CAP:
                sel = rng.choice(len(Xbg), BG_CAP, replace=False)
                Xbg, rbg = Xbg[sel], rbg[sel]
            xm = sid_te == Xs
            XX, rX = Xte[xm], rid_te[xm]
            N = held[i % len(held)]
            nm = sid_op == N
            XN, rN = Xop[nm], rid_op[nm]

            emb = lambda net, A: model.embed(net, A, device=device)
            b["raw"].append(sep_rundisjoint(XX, rX, Xbg, rbg, config.SEED + i))
            b["before"].append(sep_rundisjoint(emb(before[k]["model"], XX), rX,
                                               emb(before[k]["model"], Xbg), rbg, config.SEED + i))
            b["after"].append(sep_rundisjoint(emb(res["model"], XX), rX,
                                              emb(res["model"], Xbg), rbg, config.SEED + i))
            b["floor"].append(sep_rundisjoint(emb(res["model"], XN), rN,
                                             emb(res["model"], Xbg), rbg, config.SEED + i))

        r1, npop, r1_ci = rank1_population(Xfull, sid, rid, all_subjects, config.SEED)
        af, fl = np.array(b["after"]), np.array(b["floor"])
        m = np.isfinite(af) & np.isfinite(fl)
        try:
            p_af = float(wilcoxon(af[m], fl[m]).pvalue) if m.sum() >= 6 and np.any(af[m] != fl[m]) else 1.0
        except ValueError:
            p_af = 1.0
        row = {"channels": nch, "n_targets": len(targets),
               "after_vs_floor_wilcoxon_p": p_af,
               "after_per_subject": af[m].tolist(), "floor_per_subject": fl[m].tolist(),
               "raw_auc": float(np.nanmean(b["raw"])), "raw_std": float(np.nanstd(b["raw"])),
               "before_auc": float(np.nanmean(b["before"])), "before_std": float(np.nanstd(b["before"])),
               "after_auc": float(np.nanmean(b["after"])), "after_std": float(np.nanstd(b["after"])),
               "floor_auc": float(np.nanmean(b["floor"])), "floor_std": float(np.nanstd(b["floor"])),
               "raw_rank1": r1, "raw_rank1_ci": r1_ci, "rank1_population": npop, "rank1_chance": 1.0 / npop}
        rows.append(row)
        print(f"  c={nch:2d}ch: AUC raw={row['raw_auc']:.3f} before={row['before_auc']:.3f} "
              f"after={row['after_auc']:.3f} floor={row['floor_auc']:.3f} | "
              f"raw rank-1={r1:.3f} (chance {row['rank1_chance']:.3f})")

    out = {"protocol": "run-disjoint probe (gallery runs 2,4,6 / probe runs 8,10,12), "
                       "channel-reduction difficulty sweep", "S": S, "R": R,
           "n_targets": N_TARGETS, "rows": rows}
    with open(config.RESULTS_DIR / "reid_auc.json", "w") as f:
        json.dump(out, f, indent=2)

    xs = [r["channels"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for key, lab, col in [("raw_auc", "raw features", "#8172b3"),
                          ("before_auc", "before", "#4c72b0"),
                          ("after_auc", "after (unlearned)", "#c44e52"),
                          ("floor_auc", "never-enrolled floor", "#55a868")]:
        ys = [r[key] for r in rows]
        es = [r[key.replace("_auc", "_std")] for r in rows]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=lab, color=col)
    ax.axhline(0.5, ls="--", c="k", lw=1)
    ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels(xs)
    ax.set_xlabel("number of EEG channels"); ax.set_ylabel("re-identification AUC (run-disjoint)")
    ax.set_ylim(0.45, 1.02); ax.set_title("Re-identifiability vs channels (leakage-free)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "reid_auc_channels.png", dpi=120); plt.close(fig)

    print("-> results/reid_auc.json + figures/reid_auc_channels.png")


if __name__ == "__main__":
    main()
