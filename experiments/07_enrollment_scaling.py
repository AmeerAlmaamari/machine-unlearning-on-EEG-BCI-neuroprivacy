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
ENROLL_SIZES = [10, 20, 30, 40]
S_SISA, R = 10, 5
EXEC5, IMAG = erasure.EXEC5, erasure.IMAG


def run_seed(X, sid, rid, seed, device):
    C, E, I = erasure.three_way_split(np.unique(sid), seed)

    def msk(group, runs):
        return np.isin(sid, group) & np.isin(rid, runs)

    c_tr, c_va = msk(C, EXEC5), msk(C, [splits.EXECUTION_RUNS[5]])
    scC = StandardScaler().fit(X[c_tr])
    cls = np.unique(sid[c_tr]); lut = {int(s): i for i, s in enumerate(cls)}
    ytr = np.array([lut[int(s)] for s in sid[c_tr]])
    yva = np.array([lut[int(s)] for s in sid[c_va]])
    enc = erasure.train_encoder(scC.transform(X[c_tr]).astype(np.float32), ytr,
                                scC.transform(X[c_va]).astype(np.float32), yva,
                                len(cls), mode="cosface", device=device, seed=seed)

    i_te = msk(I, IMAG)
    I_emb = erasure.embed(enc, scC.transform(X[i_te]).astype(np.float32), device)
    sid_i = sid[i_te]

    rows = []
    for N in ENROLL_SIZES:
        E_N = E[:N]
        e_enr, e_te = msk(E_N, EXEC5), msk(E_N, IMAG)

        E_enr_emb = erasure.embed(enc, scC.transform(X[e_enr]).astype(np.float32), device)
        E_te_emb = erasure.embed(enc, scC.transform(X[e_te]).astype(np.float32), device)
        T = erasure.build_templates(E_enr_emb, sid[e_enr], E_N)
        eer_er = erasure.openset_eer_templates(T, E_te_emb, sid[e_te], I_emb, sid_i, E_N)

        CE = np.concatenate([C, E_N])
        ce_tr = np.isin(sid, CE) & np.isin(rid, EXEC5)
        ce_va = np.isin(sid, CE) & np.isin(rid, [splits.EXECUTION_RUNS[5]])
        scCE = StandardScaler().fit(X[ce_tr])
        XCEtr = scCE.transform(X[ce_tr]).astype(np.float32)
        XCEva = scCE.transform(X[ce_va]).astype(np.float32)
        XEte = scCE.transform(X[e_te]).astype(np.float32)
        XIte = scCE.transform(X[i_te]).astype(np.float32)
        sidCEtr, sidCEva, sidEte = sid[ce_tr], sid[ce_va], sid[e_te]
        classesCE = np.unique(sidCEtr)

        shards, _ = sisa.make_shards(sidCEtr, S_SISA, 1, "subject_aware", seed)
        consts, _ = unlearn.build_constituents(shards, XCEtr, sidCEtr, XCEva, sidCEva,
                                               device=device, seed=seed)
        eer_si = erasure.openset_eer_sisa(consts, "subject_aware", XEte, sidEte,
                                          XIte, classesCE, device)

        e_i = max(1, round(config.MLP_EPOCHS * 2 / (R + 1)))
        sh2, s2 = sisa.make_shards(sidCEtr, S_SISA, R, "subject_aware", seed)
        before, binfo = unlearn.build_constituents(sh2, XCEtr, sidCEtr, XCEva, sidCEva,
                                                   device=device, seed=seed,
                                                   epochs=e_i, patience=10**9)
        E_set = set(int(s) for s in E_N)
        costs = []
        for k in range(S_SISA):
            here = [int(s) for s in sh2[k]["classes"] if int(s) in E_set]
            if not here:
                continue
            res = unlearn.unlearn_subject_aware(sh2, s2, before, here[0], XCEtr, sidCEtr,
                                                XCEva, sidCEva, device=device, seed=seed,
                                                epochs=e_i, patience=10**9)
            costs.append(res["cost_steps"])

        rows.append({"n_enrolled": int(N),
                     "eer_erasure": eer_er, "eer_sisa_S10": eer_si,
                     "sisa_deletion_cost": float(np.mean(costs)),
                     "erasure_deletion_cost": 0,
                     "sisa_full_build_cost": int(binfo["cost_steps"])})
    return rows


def main() -> None:
    device = utils.get_device(); utils.set_seed()
    d = np.load(config.FEATURES_PATH)
    X, sid, rid = d["X"], d["subject_id"].astype(int), d["run_id"].astype(int)

    per_seed = [run_seed(X, sid, rid, seed, device) for seed in SEEDS]

    agg = []
    for i, N in enumerate(ENROLL_SIZES):
        er = [s[i]["eer_erasure"] for s in per_seed]
        si = [s[i]["eer_sisa_S10"] for s in per_seed]
        dc = [s[i]["sisa_deletion_cost"] for s in per_seed]
        agg.append({"n_enrolled": N,
                    "erasure_eer_mean": float(np.mean(er)), "erasure_eer_std": float(np.std(er)),
                    "sisa_eer_mean": float(np.mean(si)), "sisa_eer_std": float(np.std(si)),
                    "sisa_deletion_cost_mean": float(np.mean(dc)),
                    "erasure_deletion_cost": 0})

    out = {"protocol": "frozen 43-subject cohort encoder; enrolled population grown 10..40; "
                       "impostors = other enrolled + 20 never-enrolled",
           "seeds": SEEDS, "enroll_sizes": ENROLL_SIZES,
           "by_enrolled": agg, "per_seed": per_seed}
    with open(config.RESULTS_DIR / "enrollment_scaling.json", "w") as f:
        json.dump(out, f, indent=2)

    xs = [a["n_enrolled"] for a in agg]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2))
    a0.errorbar(xs, [a["erasure_eer_mean"] for a in agg], yerr=[a["erasure_eer_std"] for a in agg],
                marker="o", capsize=3, color="#c44e52", label="erasure by design")
    a0.errorbar(xs, [a["sisa_eer_mean"] for a in agg], yerr=[a["sisa_eer_std"] for a in agg],
                marker="s", capsize=3, color="#4c72b0", label="subject-aware SISA ($S{=}10$)")
    a0.set_xlabel("number of enrolled subjects")
    a0.set_ylabel("open-set EER (single window)")
    a0.set_title("Accuracy as the enrolled population grows")
    a0.grid(True, alpha=0.3); a0.legend(fontsize=9)

    a1.plot(xs, [a["sisa_deletion_cost_mean"] for a in agg], "s-", color="#4c72b0",
            label="subject-aware SISA ($S{=}10$)")
    a1.plot(xs, [0 for _ in agg], "o-", color="#c44e52", label="erasure by design")
    a1.set_xlabel("number of enrolled subjects")
    a1.set_ylabel("per-deletion cost (sample updates)")
    a1.set_title("Deletion cost as the enrolled population grows")
    a1.grid(True, alpha=0.3); a1.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "enrollment_scaling.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
