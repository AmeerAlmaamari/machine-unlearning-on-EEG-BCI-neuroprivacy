from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, metrics, model, sisa, splits, unlearn, utils  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
N_HOLDOUT = 20
S_GRID = [1, 5, 10, 20]
R_GRID = [1, 5, 10]
SCORE_S_GRID = [5, 10, 20]
SPEEDUP_S, SPEEDUP_R = 10, 5


def openset_eer(consts, mode, Xte, sid_te, Xop, classes, device):
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


def norm(E):
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def main() -> None:
    device = utils.get_device(); utils.set_seed()
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
    ytr = model.encode_labels(sid_tr, classes)
    yva = model.encode_labels(sid_va, classes)
    e_i = max(1, round(config.MLP_EPOCHS * 2 / (SPEEDUP_R + 1)))

    per_seed = []
    for seed in SEEDS:
        r = {"seed": seed}

        s_sweep = []
        for mode in ("subject_aware", "uniform"):
            for S in S_GRID:
                shards, _ = sisa.make_shards(sid_tr, S, 1, mode, seed)
                consts, _ = unlearn.build_constituents(shards, Xtr, sid_tr, Xva, sid_va,
                                                       device=device, seed=seed)
                eer = openset_eer(consts, mode, Xte, sid_te, Xop, classes, device)
                s_sweep.append({"mode": mode, "S": S, "open_set_eer": eer})
        r["S_sweep"] = s_sweep

        r_sweep = []
        for R in R_GRID:
            ei = max(1, round(config.MLP_EPOCHS * 2 / (R + 1)))
            shards, s2 = sisa.make_shards(sid_tr, 10, R, "subject_aware", seed)
            before, info = unlearn.build_constituents(shards, Xtr, sid_tr, Xva, sid_va,
                                                      device=device, seed=seed,
                                                      epochs=ei, patience=10**9)
            eer = openset_eer(before, "subject_aware", Xte, sid_te, Xop, classes, device)
            sample = [int(s) for k in range(10) for s in shards[k]["classes"][:2]]
            costs = [unlearn.unlearn_subject_aware(shards, s2, before, Xs, Xtr, sid_tr,
                                                   Xva, sid_va, device=device, seed=seed,
                                                   epochs=ei, patience=10**9)["cost_steps"]
                     for Xs in sample]
            r_sweep.append({"R": R, "open_set_eer": eer,
                            "mean_speedup": info["cost_steps"] / float(np.mean(costs))})
        r["R_sweep"] = r_sweep

        subsets = {"all5 (320)": list(range(config.N_FEATURES)),
                   "no_gamma (256)": [c * 5 + b for c in range(64) for b in (0, 1, 2, 3)],
                   "alpha+beta (128)": [c * 5 + b for c in range(64) for b in (2, 3)],
                   "alpha (64)": [c * 5 + 2 for c in range(64)]}
        feat = []
        for name, cols in subsets.items():
            sc = StandardScaler().fit(X[tr][:, cols])
            Xt, Xv, Xe, Xo = (sc.transform(X[m][:, cols]).astype(np.float32)
                              for m in (tr, va, te, op))
            feat.append({"features": name,
                         "open_set_eer": mono_openset_eer(Xt, ytr, Xv, yva, classes, Xe,
                                                          sid_te, Xo, device, seed)})
        r["feature_ablation"] = feat

        rng = np.random.default_rng(seed)
        probe = []
        for nb in (25, 50, 100, 200, None):
            if nb is None:
                keep = np.arange(len(sid_tr))
            else:
                keep = np.sort(np.concatenate([
                    (lambda idx: rng.choice(idx, nb, replace=False) if len(idx) > nb else idx)(
                        np.where(sid_tr == s)[0]) for s in classes]))
            eer = mono_openset_eer(Xtr[keep], ytr[keep], Xva, yva, classes, Xte, sid_te,
                                   Xop, device, seed)
            lbl = nb if nb is not None else int(np.median(np.bincount(sid_tr)[classes]))
            probe.append({"windows_per_subject": int(lbl), "open_set_eer": eer})
        r["weak_learner_probe"] = probe

        rows = []
        for S in SCORE_S_GRID:
            shards, _ = sisa.make_shards(sid_tr, S, 1, "subject_aware", seed)
            consts, _ = unlearn.build_constituents(shards, Xtr, sid_tr, Xva, sid_va,
                                                   device=device, seed=seed)
            Mt = sisa.score_matrix(consts, "subject_aware", Xte, classes, device)
            Mo = sisa.score_matrix(consts, "subject_aware", Xop, classes, device)
            soft = []
            for j, s in enumerate(classes):
                g = Mt[sid_te == s, j]
                if len(g):
                    soft.append(metrics.compute_eer(
                        g, np.concatenate([Mt[sid_te != s, j], Mo[:, j]]))[0])
            cos = []
            for k, sh in enumerate(shards):
                net = consts[k]["model"]
                Ete = norm(model.embed(net, Xte, device=device))
                Eop = norm(model.embed(net, Xop, device=device))
                for s in [int(x) for x in sh["classes"]]:
                    t = model.embed(net, Xtr[sid_tr == s], device=device).mean(0)
                    t = t / (np.linalg.norm(t) + 1e-9)
                    gen = (Ete @ t)[sid_te == s]
                    imp = np.concatenate([(Ete @ t)[sid_te != s], Eop @ t])
                    cos.append(metrics.compute_eer(gen, imp)[0])
            rows.append({"S": S, "softmax_eer": float(np.mean(soft)),
                         "cosine_eer": float(np.mean(cos))})
        r["scoring"] = rows

        shards, s2 = sisa.make_shards(sid_tr, SPEEDUP_S, SPEEDUP_R, "subject_aware", seed)
        before, binfo = unlearn.build_constituents(shards, Xtr, sid_tr, Xva, sid_va,
                                                   device=device, seed=seed,
                                                   epochs=e_i, patience=10**9)
        full = binfo["cost_steps"]
        sp = []
        for k, sh in enumerate(shards):
            for Xs in [int(s) for s in sh["classes"]]:
                res = unlearn.unlearn_subject_aware(shards, s2, before, Xs, Xtr, sid_tr,
                                                    Xva, sid_va, device=device, seed=seed,
                                                    epochs=e_i, patience=10**9)
                sp.append(full / max(res["cost_steps"], 1))
        sp = np.array(sp)
        ushards, _ = sisa.make_shards(sid_tr, SPEEDUP_S, SPEEDUP_R, "uniform", seed)
        _, uinfo = unlearn.retrain_all_minus(ushards, int(classes[0]), Xtr, sid_tr, Xva, sid_va,
                                             device=device, seed=seed,
                                             epochs=e_i, patience=10**9)
        r["speedup"] = {"median": float(np.median(sp)), "mean": float(sp.mean()),
                        "min": float(sp.min()), "max": float(sp.max()),
                        "n_subjects": int(len(sp))}
        r["cost"] = {"from_scratch": int(full), "uniform": int(uinfo["cost_steps"]),
                     "subject_aware_median": float(full / np.median(sp)),
                     "uniform_speedup": float(full / uinfo["cost_steps"])}
        per_seed.append(r)

    def ms(vals):
        a = np.array(vals, dtype=float)
        return float(a.mean()), float(a.std())

    agg = {"seeds": SEEDS, "per_seed": per_seed}

    s_sweep = []
    for mode in ("subject_aware", "uniform"):
        for S in S_GRID:
            v = [next(x["open_set_eer"] for x in r["S_sweep"]
                      if x["mode"] == mode and x["S"] == S) for r in per_seed]
            m, s = ms(v)
            s_sweep.append({"mode": mode, "S": S, "eer_mean": m, "eer_std": s})
    agg["S_sweep"] = s_sweep

    r_sweep = []
    for R in R_GRID:
        e = [next(x["open_set_eer"] for x in r["R_sweep"] if x["R"] == R) for r in per_seed]
        sp_ = [next(x["mean_speedup"] for x in r["R_sweep"] if x["R"] == R) for r in per_seed]
        em, es = ms(e); sm, ss = ms(sp_)
        r_sweep.append({"R": R, "eer_mean": em, "eer_std": es,
                        "speedup_mean": sm, "speedup_std": ss})
    agg["R_sweep"] = r_sweep

    names = [f["features"] for f in per_seed[0]["feature_ablation"]]
    agg["feature_ablation"] = []
    for n in names:
        v = [next(x["open_set_eer"] for x in r["feature_ablation"] if x["features"] == n)
             for r in per_seed]
        m, s = ms(v)
        agg["feature_ablation"].append({"features": n, "eer_mean": m, "eer_std": s})

    budgets = [p["windows_per_subject"] for p in per_seed[0]["weak_learner_probe"]]
    agg["weak_learner_probe"] = []
    for b in budgets:
        v = [next(x["open_set_eer"] for x in r["weak_learner_probe"]
                  if x["windows_per_subject"] == b) for r in per_seed]
        m, s = ms(v)
        agg["weak_learner_probe"].append({"windows_per_subject": b, "eer_mean": m, "eer_std": s})

    agg["scoring"] = []
    for S in SCORE_S_GRID:
        so = [next(x["softmax_eer"] for x in r["scoring"] if x["S"] == S) for r in per_seed]
        co = [next(x["cosine_eer"] for x in r["scoring"] if x["S"] == S) for r in per_seed]
        sm, ss = ms(so); cm, cs = ms(co)
        agg["scoring"].append({"S": S, "softmax_mean": sm, "softmax_std": ss,
                               "cosine_mean": cm, "cosine_std": cs})
    sd = {x["S"]: x for x in agg["scoring"]}
    agg["scoring_rise_5_to_20"] = {
        "softmax": sd[20]["softmax_mean"] - sd[5]["softmax_mean"],
        "cosine": sd[20]["cosine_mean"] - sd[5]["cosine_mean"]}

    for key in ("median", "mean", "min", "max"):
        m, s = ms([r["speedup"][key] for r in per_seed])
        agg.setdefault("speedup", {})[key + "_mean"] = m
        agg["speedup"][key + "_std"] = s
    for key in ("from_scratch", "uniform", "subject_aware_median", "uniform_speedup"):
        m, s = ms([r["cost"][key] for r in per_seed])
        agg.setdefault("cost", {})[key + "_mean"] = m
        agg["cost"][key + "_std"] = s

    with open(config.RESULTS_DIR / "sweeps.json", "w") as f:
        json.dump(agg, f, indent=2)


if __name__ == "__main__":
    main()
