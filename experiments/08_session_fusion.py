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
from src import config, erasure, metrics, sisa, splits, unlearn, utils  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
K_GRID = [1, 5, 15, 30, 0]
EXEC5, IMAG = erasure.EXEC5, erasure.IMAG


def fuse(scores, groups, widx, K):
    out = []
    order = np.lexsort((widx, groups))
    scores, groups = scores[order], groups[order]
    for g in np.unique(groups):
        s = scores[groups == g]
        if K == 0 or K >= len(s):
            out.append(s.mean())
        else:
            n = (len(s) // K) * K
            if n == 0:
                out.append(s.mean())
            else:
                out.append(s[:n].reshape(-1, K).mean(1))
    return np.concatenate([np.atleast_1d(o) for o in out])


def eer_at_K(gen_s, gen_grp, gen_w, imp_s, imp_grp, imp_w, K):
    g = fuse(gen_s, gen_grp, gen_w, K)
    i = fuse(imp_s, imp_grp, imp_w, K)
    return metrics.compute_eer(g, i)[0]


def run_seed(X, sid, rid, widx, seed, device):
    C, E, I = erasure.three_way_split(np.unique(sid), seed)

    def msk(group, runs):
        return np.isin(sid, group) & np.isin(rid, runs)

    c_tr = msk(C, EXEC5); c_va = msk(C, [splits.EXECUTION_RUNS[5]])
    scC = StandardScaler().fit(X[c_tr])
    cls = np.unique(sid[c_tr]); lut = {int(s): i for i, s in enumerate(cls)}
    ytr = np.array([lut[int(s)] for s in sid[c_tr]])
    yva = np.array([lut[int(s)] for s in sid[c_va]])
    enc = erasure.train_encoder(scC.transform(X[c_tr]).astype(np.float32), ytr,
                                scC.transform(X[c_va]).astype(np.float32), yva,
                                len(cls), mode="cosface", device=device, seed=seed)
    e_enr = msk(E, EXEC5)
    E_enr_emb = erasure.embed(enc, scC.transform(X[e_enr]).astype(np.float32), device)
    T = erasure.build_templates(E_enr_emb, sid[e_enr], E)
    te = msk(np.concatenate([E, I]), IMAG)
    Xte_emb = erasure.embed(enc, scC.transform(X[te]).astype(np.float32), device)
    sid_te, widx_te = sid[te], widx[te]

    CE = np.concatenate([C, E])
    ce_tr = np.isin(sid, CE) & np.isin(rid, EXEC5)
    ce_va = np.isin(sid, CE) & np.isin(rid, [splits.EXECUTION_RUNS[5]])
    scCE = StandardScaler().fit(X[ce_tr])
    XCEtr = scCE.transform(X[ce_tr]).astype(np.float32)
    XCEva = scCE.transform(X[ce_va]).astype(np.float32)
    Xte_soft = scCE.transform(X[te]).astype(np.float32)
    sidCEtr, sidCEva = sid[ce_tr], sid[ce_va]
    classesCE = np.unique(sidCEtr)
    col = {int(s): j for j, s in enumerate(classesCE)}

    soft = {}
    for name, S in [("monolithic", 1), ("sisa_S10", 10)]:
        shards, _ = sisa.make_shards(sidCEtr, S, 1, "subject_aware", seed)
        consts, _ = unlearn.build_constituents(shards, XCEtr, sidCEtr, XCEva, sidCEva,
                                               device=device, seed=seed)
        soft[name] = sisa.score_matrix(consts, "subject_aware", Xte_soft, classesCE, device)

    methods = {}
    grp_all = sid_te.astype(np.int64) * 100 + rid[te].astype(np.int64)
    for name in ("erasure", "monolithic", "sisa_S10"):
        per_K = {}
        for K in K_GRID:
            eers = []
            for u in E:
                if name == "erasure":
                    scv = Xte_emb @ T[int(u)]
                else:
                    scv = soft[name][:, col[int(u)]]
                gen_m = sid_te == u
                imp_m = sid_te != u
                eers.append(eer_at_K(scv[gen_m], grp_all[gen_m], widx_te[gen_m],
                                     scv[imp_m], grp_all[imp_m], widx_te[imp_m], K))
            per_K[K] = float(np.mean(eers))
        methods[name] = per_K
    return {"cohort": C.tolist(), "enrolled": E.tolist(), "impostor": I.tolist(),
            "eer_by_K": methods}


def main() -> None:
    device = utils.get_device(); utils.set_seed()
    d = np.load(config.FEATURES_PATH)
    X = d["X"]
    sid = d["subject_id"].astype(int)
    rid = d["run_id"].astype(int)
    widx = d["window_idx"].astype(int)

    per_seed = [run_seed(X, sid, rid, widx, seed, device) for seed in SEEDS]

    agg = {}
    for m in ("erasure", "sisa_S10", "monolithic"):
        agg[m] = {}
        for K in K_GRID:
            vals = [r["eer_by_K"][m][K] for r in per_seed]
            agg[m][str(K)] = {"eer_mean": float(np.mean(vals)), "eer_std": float(np.std(vals))}
    summary = {"protocol": "session-level fusion, blocks of K windows within a run, "
                           "3-way split, cross-task open-set", "seeds": SEEDS,
               "K_grid": K_GRID, "windows_are_2s_50pct_overlap": True,
               "eer_by_K": agg, "per_seed": per_seed}
    with open(config.RESULTS_DIR / "session_fusion.json", "w") as f:
        json.dump(summary, f, indent=2)

    PLOT_K = [1, 5, 15, 30]
    labels = {1: "1 (2s)", 5: "5 (~6s)", 15: "15 (~16s)", 30: "30 (~31s)"}
    xs = list(range(len(PLOT_K)))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    styles = {"erasure": ("#c44e52", "erasure-by-design"),
              "sisa_S10": ("#4c72b0", "SISA $S{=}10$"),
              "monolithic": ("#7f7f7f", "monolithic")}
    for m, (col, lab) in styles.items():
        ys = [agg[m][str(K)]["eer_mean"] for K in PLOT_K]
        es = [agg[m][str(K)]["eer_std"] for K in PLOT_K]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color=col, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([labels[k] for k in PLOT_K])
    ax.set_xlabel("fused windows per authentication (approx. seconds)")
    ax.set_ylabel("open-set EER (log scale)")
    ax.set_title("Session-level verification")
    ax.grid(True, alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "session_fusion.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
