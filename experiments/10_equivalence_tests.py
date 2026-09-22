from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

ALPHA = 0.05
N_TESTS = 4


def analyse(after, floor, margin):
    after = np.asarray(after, float)
    floor = np.asarray(floor, float)
    m_ = np.isfinite(after) & np.isfinite(floor)
    d = after[m_] - floor[m_]
    n = len(d)
    m = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    df = n - 1
    ci95 = (m - stats.t.ppf(0.975, df) * se, m + stats.t.ppf(0.975, df) * se)
    ci90 = (m - stats.t.ppf(0.95, df) * se, m + stats.t.ppf(0.95, df) * se)
    p_low = 1 - stats.t.cdf((m + margin) / se, df)
    p_up = stats.t.cdf((m - margin) / se, df)
    tost_p = max(p_low, p_up)
    equiv = (ci90[0] > -margin) and (ci90[1] < margin)
    p_after_greater = 1 - stats.t.cdf(m / se, df)
    return {"n": n, "mean_diff": m, "sd_diff": sd,
            "ci95": [float(ci95[0]), float(ci95[1])],
            "ci90": [float(ci90[0]), float(ci90[1])],
            "margin": margin, "tost_p": float(tost_p), "equivalent": bool(equiv),
            "p_after_greater_than_floor": float(p_after_greater),
            "after_not_above_floor": bool(m <= 0 or ci95[1] < margin)}


def run_metric(rows, after_key, floor_key, margin):
    out = []
    for r in rows:
        res = analyse(r[after_key], r[floor_key], margin)
        res["channels"] = r["channels"]
        out.append(res)
    return out


def main() -> None:
    rows = json.load(open(config.RESULTS_DIR / "reid_audit.json"))["rows"]

    auc = run_metric(rows, "auc_after_per_subject", "auc_floor_per_subject", 0.02)
    rank1 = run_metric(rows, "r1_after_per_subject", "r1_floor_per_subject", 0.10)

    bonf = ALPHA / N_TESTS
    out = {"alpha": ALPHA, "bonferroni_threshold": bonf, "n_tests_per_metric": N_TESTS,
           "auc": {"margin": 0.02, "per_channel": auc,
                   "equivalent_all_channels": all(a["tost_p"] < ALPHA for a in auc),
                   "equivalent_all_channels_bonferroni": all(a["tost_p"] < bonf for a in auc)},
           "rank1": {"margin": 0.10, "per_channel": rank1,
                     "after_not_above_floor_all_channels":
                         all(r["after_not_above_floor"] for r in rank1)}}
    with open(config.RESULTS_DIR / "equivalence.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
