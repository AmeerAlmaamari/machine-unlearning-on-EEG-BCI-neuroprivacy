from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model, sisa, splits, unlearn, utils  # noqa: E402

S, R = 10, 5
N_HOLDOUT = 20
CHANNELS = [64, 16, 8, 4]
GAL_RUNS, PROBE_RUNS = [2, 4, 6], [8, 10, 12]
BG_CAP = 3000


def select_cols(c):
    idx = np.unique(np.linspace(0, 63, c).astype(int))
    return [ch * 5 + b for ch in idx for b in range(5)], len(idx)


def _norm(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)


def sep_rundisjoint(Xt, rt, Xb, rb, seed):
    rng = np.random.default_rng(seed)

    def part(runs):
        A, B = Xt[np.isin(rt, runs)], Xb[np.isin(rb, runs)]
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


def templates(feat, sid_g, subjects):
    T, tids = [], []
    for s in subjects:
        g = feat[sid_g == s]
        if len(g):
            T.append(g.mean(0)); tids.append(int(s))
    return _norm(np.asarray(T, dtype=np.float64)), np.asarray(tids)


def rank1_target(gal_feat, gal_sid, subjects, probe_feat, target_id):
    if len(probe_feat) == 0:
        return float("nan")
    T, tids = templates(gal_feat, gal_sid, subjects)
    P = _norm(np.asarray(probe_feat, dtype=np.float64))
    return float(np.mean(tids[np.argmax(P @ T.T, axis=1)] == int(target_id)))


def main() -> None:
    device = utils.get_device(); utils.set_seed()
    d = np.load(config.FEATURES_PATH)
    Xall, sid, rid = d["X"], d["subject_id"].astype(int), d["run_id"].astype(int)
    enrolled, held = splits.holdout_subjects(N_HOLDOUT, config.SEED)
    EXEC, IMAG = splits.EXECUTION_RUNS, splits.IMAGERY_RUNS
    tr = np.isin(sid, enrolled) & np.isin(rid, EXEC[:5])
    va = np.isin(sid, enrolled) & np.isin(rid, EXEC[5:])
    te = np.isin(sid, enrolled) & np.isin(rid, IMAG)
    op = np.isin(sid, held) & np.isin(rid, IMAG)
    all_subjects = np.array(sorted(set(enrolled) | set(held)))
    gal, prb = np.isin(rid, GAL_RUNS), np.isin(rid, PROBE_RUNS)

    rows = []
    for c in CHANNELS:
        cols, nch = select_cols(c)
        Xc = Xall[:, cols]
        scaler = StandardScaler().fit(Xc[tr])
        Xtr = scaler.transform(Xc[tr]).astype(np.float32)
        Xva = scaler.transform(Xc[va]).astype(np.float32)
        Xte = scaler.transform(Xc[te]).astype(np.float32)
        Xop = scaler.transform(Xc[op]).astype(np.float32)
        Xsc = scaler.transform(Xc).astype(np.float32)
        sid_tr, sid_va = sid[tr], sid[va]
        sid_te, rid_te = sid[te], rid[te]
        sid_op, rid_op = sid[op], rid[op]

        Xgal_raw, sgal = Xsc[gal], sid[gal]
        Xprb_raw, sprb = Xsc[prb], sid[prb]

        shards, s2 = sisa.make_shards(sid_tr, S, R, "subject_aware", config.SEED)
        before, _ = unlearn.build_constituents(shards, Xtr, sid_tr, Xva, sid_va,
                                               device=device, seed=config.SEED)
        emb = lambda net, A: model.embed(net, A, device=device)
        before_gal_emb = {}

        targets = [int(s) for s in np.unique(sid_tr)]
        b = {k: [] for k in ("auc_raw", "auc_before", "auc_after", "auc_floor",
                             "r1_raw", "r1_before", "r1_after", "r1_floor")}
        rng = np.random.default_rng(config.SEED)
        for i, Xs in enumerate(targets):
            k = s2[Xs]
            res = unlearn.unlearn_subject_aware(shards, s2, before, Xs, Xtr, sid_tr,
                                                Xva, sid_va, device=device, seed=config.SEED)
            after_net, before_net = res["model"], before[k]["model"]
            N = int(held[i % len(held)])

            bg_sub = np.array([s for s in enrolled if s != Xs])
            bgm = np.isin(sid_te, bg_sub)
            Xbg, rbg = Xte[bgm], rid_te[bgm]
            if len(Xbg) > BG_CAP:
                sel = rng.choice(len(Xbg), BG_CAP, replace=False)
                Xbg, rbg = Xbg[sel], rbg[sel]
            xm = sid_te == Xs
            XX, rX = Xte[xm], rid_te[xm]
            nm = sid_op == N
            XN, rN = Xop[nm], rid_op[nm]
            sd = config.SEED + i
            b["auc_raw"].append(sep_rundisjoint(XX, rX, Xbg, rbg, sd))
            b["auc_before"].append(sep_rundisjoint(emb(before_net, XX), rX,
                                                   emb(before_net, Xbg), rbg, sd))
            b["auc_after"].append(sep_rundisjoint(emb(after_net, XX), rX,
                                                  emb(after_net, Xbg), rbg, sd))
            b["auc_floor"].append(sep_rundisjoint(emb(after_net, XN), rN,
                                                  emb(after_net, Xbg), rbg, sd))

            if k not in before_gal_emb:
                before_gal_emb[k] = emb(before_net, Xgal_raw)
            bg_emb = before_gal_emb[k]
            ag_emb = emb(after_net, Xgal_raw)
            Xprb_X = Xprb_raw[sprb == Xs]
            Xprb_N = Xprb_raw[sprb == N]
            b["r1_raw"].append(rank1_target(Xgal_raw, sgal, all_subjects, Xprb_X, Xs))
            b["r1_before"].append(rank1_target(bg_emb, sgal, all_subjects,
                                               emb(before_net, Xprb_X), Xs))
            b["r1_after"].append(rank1_target(ag_emb, sgal, all_subjects,
                                              emb(after_net, Xprb_X), Xs))
            b["r1_floor"].append(rank1_target(ag_emb, sgal, all_subjects,
                                              emb(after_net, Xprb_N), N))

        row = {"channels": nch, "n_targets": len(targets), "n_floor_refs": len(held),
               "population": int(len(all_subjects)), "chance": 1.0 / len(all_subjects)}
        for key, vals in b.items():
            a = np.array(vals, dtype=float)
            a = a[np.isfinite(a)]
            row[f"{key}_mean"] = float(a.mean())
            row[f"{key}_std"] = float(a.std())
        row["auc_after_per_subject"] = b["auc_after"]
        row["auc_floor_per_subject"] = b["auc_floor"]
        row["r1_after_per_subject"] = b["r1_after"]
        row["r1_floor_per_subject"] = b["r1_floor"]
        rows.append(row)

    out = {"protocol": "full-scale run-disjoint audit at the headline configuration; "
                       "all enrolled subjects removed in turn; all 20 never-enrolled "
                       "subjects rotated as the floor",
           "S": S, "R": R, "rows": rows}
    with open(config.RESULTS_DIR / "reid_audit.json", "w") as f:
        json.dump(out, f, indent=2)

    xs = [r["channels"] for r in rows]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, lab, col in [("r1_raw", "raw features", "#8172b3"),
                          ("r1_before", "before", "#4c72b0"),
                          ("r1_after", "after (unlearned)", "#c44e52"),
                          ("r1_floor", "never-enrolled floor", "#55a868")]:
        a0.errorbar(xs, [r[f"{key}_mean"] for r in rows], yerr=[r[f"{key}_std"] for r in rows],
                    marker="o", capsize=3, label=lab, color=col)
    a0.axhline(rows[0]["chance"], ls="--", c="k", lw=1, label="chance (1/103)")
    a0.set_xscale("log", base=2); a0.set_xticks(xs); a0.set_xticklabels(xs)
    a0.set_xlabel("number of EEG channels")
    a0.set_ylabel("rank-1 identification over 103 subjects")
    a0.set_title("Model-embedding re-identification")
    a0.grid(True, alpha=0.3); a0.legend(fontsize=8)

    for key, lab, col in [("auc_raw", "raw features", "#8172b3"),
                          ("auc_before", "before", "#4c72b0"),
                          ("auc_after", "after (unlearned)", "#c44e52"),
                          ("auc_floor", "never-enrolled floor", "#55a868")]:
        a1.errorbar(xs, [r[f"{key}_mean"] for r in rows], yerr=[r[f"{key}_std"] for r in rows],
                    marker="o", capsize=3, label=lab, color=col)
    a1.axhline(0.5, ls="--", c="k", lw=1)
    a1.set_xscale("log", base=2); a1.set_xticks(xs); a1.set_xticklabels(xs)
    a1.set_xlabel("number of EEG channels")
    a1.set_ylabel("separability AUC (run-disjoint)")
    a1.set_ylim(0.45, 1.02); a1.set_title("After removal tracks the never-enrolled floor")
    a1.grid(True, alpha=0.3); a1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "reidentification_audit.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
