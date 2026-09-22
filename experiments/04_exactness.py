from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model, sisa, splits, unlearn, utils  # noqa: E402

S, R, N_HOLDOUT = 10, 5, 20


def main() -> None:
    device = utils.get_device(); utils.set_seed()
    d = np.load(config.FEATURES_PATH)
    X, sid, rid = d["X"], d["subject_id"].astype(int), d["run_id"]
    enrolled, _ = splits.holdout_subjects(N_HOLDOUT, config.SEED)
    EXEC, IMAG = splits.EXECUTION_RUNS, splits.IMAGERY_RUNS
    tr = np.isin(sid, enrolled) & np.isin(rid, EXEC[:5])
    va = np.isin(sid, enrolled) & np.isin(rid, EXEC[5:])
    te = np.isin(sid, enrolled) & np.isin(rid, IMAG)
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xva, Xte = (scaler.transform(X[m]).astype(np.float32) for m in (tr, va, te))
    sid_tr, sid_va = sid[tr], sid[va]

    shards, s2 = sisa.make_shards(sid_tr, S, R, "subject_aware", config.SEED)
    before, _ = unlearn.build_constituents(shards, Xtr, sid_tr, Xva, sid_va,
                                           device=device, seed=config.SEED)

    targets = [int(s) for s in np.unique(sid_tr)]
    per, pdiffs, probdiffs = [], [], []
    for Xs in targets:
        k = s2[Xs]
        res = unlearn.unlearn_subject_aware(shards, s2, before, Xs, Xtr, sid_tr,
                                            Xva, sid_va, device=device, seed=config.SEED)
        fs = unlearn.train_shard_sliced(shards[k], Xtr, sid_tr, Xva, sid_va, exclude=Xs,
                                        start_slice=0, init_state=None, device=device,
                                        seed=config.SEED)
        md = max(float((a - b).abs().max()) for a, b in
                 zip(res["model"].state_dict().values(), fs["model"].state_dict().values()))
        pd = float(np.mean(np.abs(model.predict_proba(res["model"], Xte, device=device)
                                  - model.predict_proba(fs["model"], Xte, device=device))))
        pdiffs.append(md); probdiffs.append(pd)
        per.append({"subject": Xs, "shard": int(k), "slice": int(res["slice"]),
                    "max_param_diff": md, "mean_prob_diff": pd})

    out = {"config": {"S": S, "R": R}, "n_subjects": len(targets),
           "max_param_diff": float(np.max(pdiffs)),
           "mean_param_diff": float(np.mean(pdiffs)),
           "max_prob_diff": float(np.max(probdiffs)),
           "mean_prob_diff": float(np.mean(probdiffs)),
           "n_exactly_identical": int(sum(1 for p in pdiffs if p == 0.0)),
           "bitwise_identical_all": all(p == 0.0 for p in pdiffs),
           "per_subject": per}
    with open(config.RESULTS_DIR / "exactness.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
