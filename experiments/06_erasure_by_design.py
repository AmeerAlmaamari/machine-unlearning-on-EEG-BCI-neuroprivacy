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
from src import config, erasure, sisa, splits, unlearn, utils  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
EXEC5, IMAG = erasure.EXEC5, erasure.IMAG


def run_seed(X, sid, rid, seed, device):
    C, E, I = erasure.three_way_split(np.unique(sid), seed)

    def m(group, runs):
        return np.isin(sid, group) & np.isin(rid, runs)

    c_tr = m(C, EXEC5); c_va = m(C, [splits.EXECUTION_RUNS[5]])
    scaler = StandardScaler().fit(X[c_tr])

    def sc(mask):
        return scaler.transform(X[mask]).astype(np.float32)

    e_enr = m(E, EXEC5); e_te = m(E, IMAG); i_te = m(I, IMAG)
    Xc_tr, yc_tr = sc(c_tr), sid[c_tr]
    Xc_va, yc_va = sc(c_va), sid[c_va]

    out = {"cohort": C.tolist(), "enrolled": E.tolist(), "impostor": I.tolist()}

    sweep = []
    for k in erasure.COHORT_SWEEP:
        Ck = C[:k]
        km = np.isin(sid, Ck)
        tr = km & np.isin(rid, EXEC5)
        va = km & np.isin(rid, [splits.EXECUTION_RUNS[5]])
        sck = StandardScaler().fit(X[tr])
        Xtr = sck.transform(X[tr]).astype(np.float32)
        Xva = sck.transform(X[va]).astype(np.float32)
        cls = np.unique(sid[tr]); lut = {int(s): i for i, s in enumerate(cls)}
        ytr = np.array([lut[int(s)] for s in sid[tr]])
        yva = np.array([lut[int(s)] for s in sid[va]])
        enc = erasure.train_encoder(Xtr, ytr, Xva, yva, len(cls), mode="cosface",
                                    device=device, seed=seed)
        E_enr_emb = erasure.embed(enc, sck.transform(X[e_enr]).astype(np.float32), device)
        E_te_emb = erasure.embed(enc, sck.transform(X[e_te]).astype(np.float32), device)
        I_te_emb = erasure.embed(enc, sck.transform(X[i_te]).astype(np.float32), device)
        C_emb = erasure.embed(enc, Xva, device)
        T = erasure.build_templates(E_enr_emb, sid[e_enr], E)
        eer = erasure.openset_eer_templates(T, E_te_emb, sid[e_te], I_te_emb, sid[i_te], E)
        eer_sn = erasure.openset_eer_templates(T, E_te_emb, sid[e_te], I_te_emb, sid[i_te], E,
                                               snorm_cohort=C_emb)
        sweep.append({"cohort_size": int(k), "eer": eer, "eer_snorm": eer_sn})
    out["cohort_sweep"] = sweep

    cls = np.unique(yc_tr); lut = {int(s): i for i, s in enumerate(cls)}
    ytr = np.array([lut[int(s)] for s in yc_tr]); yva = np.array([lut[int(s)] for s in yc_va])
    abl = {}
    for mode in ("cosface", "ce"):
        enc = erasure.train_encoder(Xc_tr, ytr, Xc_va, yva, len(cls), mode=mode,
                                    device=device, seed=seed)
        E_enr_emb = erasure.embed(enc, sc(e_enr), device)
        E_te_emb = erasure.embed(enc, sc(e_te), device)
        I_te_emb = erasure.embed(enc, sc(i_te), device)
        C_emb = erasure.embed(enc, Xc_va, device)
        T = erasure.build_templates(E_enr_emb, sid[e_enr], E)
        eer = erasure.openset_eer_templates(T, E_te_emb, sid[e_te], I_te_emb, sid[i_te], E)
        eer_sn = erasure.openset_eer_templates(T, E_te_emb, sid[e_te], I_te_emb, sid[i_te], E,
                                               snorm_cohort=C_emb)
        abl[mode] = {"eer": eer, "eer_snorm": eer_sn}
    out["encoder_ablation"] = abl
    out["erasure_full"] = abl["cosface"]

    CE = np.concatenate([C, E])
    ce_tr = np.isin(sid, CE) & np.isin(rid, EXEC5)
    ce_va = np.isin(sid, CE) & np.isin(rid, [splits.EXECUTION_RUNS[5]])
    scCE = StandardScaler().fit(X[ce_tr])
    XCEtr = scCE.transform(X[ce_tr]).astype(np.float32)
    XCEva = scCE.transform(X[ce_va]).astype(np.float32)
    XEte = scCE.transform(X[e_te]).astype(np.float32)
    XIte = scCE.transform(X[i_te]).astype(np.float32)
    sidCEtr, sidCEva, sidEte = sid[ce_tr], sid[ce_va], sid[e_te]
    classesCE = np.unique(sidCEtr)
    E_set = set(int(s) for s in E)

    baselines = {}
    for name, S in [("monolithic", 1), ("sisa_S5", 5), ("sisa_S10", 10), ("sisa_S20", 20)]:
        shards, _ = sisa.make_shards(sidCEtr, S, 1, "subject_aware", seed)
        consts, _ = unlearn.build_constituents(shards, XCEtr, sidCEtr, XCEva, sidCEva,
                                               device=device, seed=seed)
        eer = erasure.openset_eer_sisa(consts, "subject_aware", XEte, sidEte, XIte,
                                       classesCE, device)
        baselines[name] = {"S": S, "eer": eer}
    out["baselines"] = baselines

    R = 5
    e_i = max(1, round(config.MLP_EPOCHS * 2 / (R + 1)))
    shards, s2 = sisa.make_shards(sidCEtr, 10, R, "subject_aware", seed)
    before, binfo = unlearn.build_constituents(shards, XCEtr, sidCEtr, XCEva, sidCEva,
                                               device=device, seed=seed,
                                               epochs=e_i, patience=10**9)
    full_cost = binfo["cost_steps"]
    del_costs = []
    for k in range(10):
        here = [int(s) for s in shards[k]["classes"] if int(s) in E_set]
        if not here:
            continue
        res = unlearn.unlearn_subject_aware(shards, s2, before, here[0], XCEtr, sidCEtr,
                                            XCEva, sidCEva, device=device, seed=seed,
                                            epochs=e_i, patience=10**9)
        del_costs.append(res["cost_steps"])
    out["cost"] = {"full_build_cost": int(full_cost),
                   "monolithic_deletion_cost": int(full_cost),
                   "sisa_S10_deletion_cost": float(np.mean(del_costs)),
                   "sisa_S10_speedup": float(full_cost / np.mean(del_costs)),
                   "erasure_deletion_cost": 0}
    return out


def main() -> None:
    device = utils.get_device(); utils.set_seed()
    d = np.load(config.FEATURES_PATH)
    X, sid, rid = d["X"], d["subject_id"].astype(int), d["run_id"].astype(int)

    per_seed = [run_seed(X, sid, rid, seed, device) for seed in SEEDS]

    def agg(path):
        vals = []
        for r in per_seed:
            node = r
            for p in path:
                node = node[p]
            vals.append(node)
        return float(np.mean(vals)), float(np.std(vals))

    summary = {"protocol": "3-way split (cohort/enrolled/impostor), cross-task, open-set",
               "sizes": {"cohort": erasure.N_COHORT, "enrolled": erasure.N_ENROLLED,
                         "impostor": erasure.N_IMPOSTOR},
               "seeds": SEEDS, "per_seed": per_seed}

    sweep_agg = []
    for i, k in enumerate(erasure.COHORT_SWEEP):
        e = [r["cohort_sweep"][i]["eer"] for r in per_seed]
        esn = [r["cohort_sweep"][i]["eer_snorm"] for r in per_seed]
        sweep_agg.append({"cohort_size": k,
                          "eer_mean": float(np.mean(e)), "eer_std": float(np.std(e)),
                          "eer_snorm_mean": float(np.mean(esn)), "eer_snorm_std": float(np.std(esn))})
    summary["cohort_sweep"] = sweep_agg

    summary["erasure_full"] = {
        "eer_mean": agg(["erasure_full", "eer"])[0], "eer_std": agg(["erasure_full", "eer"])[1],
        "eer_snorm_mean": agg(["erasure_full", "eer_snorm"])[0],
        "eer_snorm_std": agg(["erasure_full", "eer_snorm"])[1]}
    summary["encoder_ablation"] = {
        mode: {"eer_mean": agg(["encoder_ablation", mode, "eer"])[0],
               "eer_std": agg(["encoder_ablation", mode, "eer"])[1]}
        for mode in ("cosface", "ce")}
    summary["baselines"] = {
        b: {"eer_mean": agg(["baselines", b, "eer"])[0], "eer_std": agg(["baselines", b, "eer"])[1]}
        for b in ("monolithic", "sisa_S5", "sisa_S10", "sisa_S20")}
    summary["cost"] = {
        "full_build_cost": agg(["cost", "full_build_cost"])[0],
        "sisa_S10_deletion_cost": agg(["cost", "sisa_S10_deletion_cost"])[0],
        "sisa_S10_speedup": agg(["cost", "sisa_S10_speedup"])[0],
        "erasure_deletion_cost": 0}

    with open(config.RESULTS_DIR / "erasure_by_design.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ks = [s["cohort_size"] for s in sweep_agg]
    ax.errorbar(ks, [s["eer_mean"] for s in sweep_agg], yerr=[s["eer_std"] for s in sweep_agg],
                marker="o", capsize=3, color="#c44e52", label="cosine template")
    ax.errorbar(ks, [s["eer_snorm_mean"] for s in sweep_agg], yerr=[s["eer_snorm_std"] for s in sweep_agg],
                marker="s", capsize=3, color="#55a868", label="+ cohort s-norm")
    ax.axhline(summary["baselines"]["monolithic"]["eer_mean"], ls="--", c="#7f7f7f",
               label="monolithic (train on users)")
    ax.set_xlabel("number of cohort identities training the encoder")
    ax.set_ylabel("open-set EER (enrolled users)")
    ax.set_title("Generalization of the cohort encoder")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "cohort_sweep.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
