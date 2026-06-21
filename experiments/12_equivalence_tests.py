"""Equivalence and non-superiority tests for "after removal == never-enrolled".

Non-rejection of a null is not evidence of equivalence, so this script reads the
per-subject paired measurements produced by the re-identification audits and
computes, for the after-vs-floor contrast:

  - paired mean difference d = after - floor, with a two-sided 95% t CI;
  - a TOST equivalence test (two one-sided tests) at a pre-specified margin,
    equivalence declared when the 90% CI of d lies inside [-margin, +margin];
  - a one-sided non-superiority check that the after-model does not identify the
    removed subject more than a never-enrolled subject (upper 95% bound of d).

Two metrics are analysed:
  - separability AUC      from results/reid_auc.json    (margin 0.02)
  - model-embedding rank-1 from results/reid_rank1.json (margin 0.10)

Four channel counts are tested per metric, so with a family of four the 0.05
threshold becomes 0.0125 (Bonferroni). Both raw and corrected verdicts are printed.

Run:
    python experiments/12_equivalence_tests.py

Outputs:
    results/equivalence_tests.json

Reads:
    results/reid_auc.json, results/reid_rank1.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

ALPHA = 0.05
N_TESTS = 4  # four channel counts per metric (Bonferroni family)


def analyse(after, floor, margin):
    after = np.asarray(after, float)
    floor = np.asarray(floor, float)
    d = after - floor
    n = len(d)
    m = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    df = n - 1
    # two-sided 95% CI
    tcrit95 = stats.t.ppf(0.975, df)
    ci95 = (m - tcrit95 * se, m + tcrit95 * se)
    # 90% CI (for TOST at alpha=0.05)
    tcrit90 = stats.t.ppf(0.95, df)
    ci90 = (m - tcrit90 * se, m + tcrit90 * se)
    # TOST: H0a diff <= -margin ; H0b diff >= +margin
    t_low = (m + margin) / se
    p_low = 1 - stats.t.cdf(t_low, df)
    t_up = (m - margin) / se
    p_up = stats.t.cdf(t_up, df)
    tost_p = max(p_low, p_up)
    equiv = (ci90[0] > -margin) and (ci90[1] < margin)
    # one-sided non-superiority: is after greater than floor? p for H1: d>0
    p_after_greater = 1 - stats.t.cdf(m / se, df)
    return {"n": n, "mean_diff": m, "sd_diff": sd,
            "ci95": [float(ci95[0]), float(ci95[1])],
            "ci90": [float(ci90[0]), float(ci90[1])],
            "margin": margin, "tost_p": float(tost_p), "equivalent": bool(equiv),
            "p_after_greater_than_floor": float(p_after_greater),
            "after_not_above_floor": bool(m <= 0 or ci95[1] < margin)}


def run_metric(rows, after_key, floor_key, ch_key, margin, label):
    print(f"\n=== {label} (margin {margin}) ===")
    out = []
    for r in rows:
        res = analyse(r[after_key], r[floor_key], margin)
        res["channels"] = r[ch_key]
        out.append(res)
        eq = "EQUIV" if res["equivalent"] else "not-equiv"
        ns = "after<=floor" if res["after_not_above_floor"] else "after>floor"
        print(f"  {r[ch_key]:2d}ch: d={res['mean_diff']:+.3f} "
              f"95%CI[{res['ci95'][0]:+.3f},{res['ci95'][1]:+.3f}] "
              f"TOST p={res['tost_p']:.3f} [{eq}] | {ns} "
              f"(p_after>floor={res['p_after_greater_than_floor']:.3f})")
    return out


def main() -> None:
    p_auc = json.load(open(config.RESULTS_DIR / "reid_auc.json"))
    p_rank1 = json.load(open(config.RESULTS_DIR / "reid_rank1.json"))

    auc = run_metric(p_auc["rows"], "after_per_subject", "floor_per_subject",
                     "channels", 0.02, "Separability AUC")
    rank1 = run_metric(p_rank1["rows"], "after_per_subject", "floor_per_subject",
                       "channels", 0.10, "Model-embedding rank-1")

    bonf = ALPHA / N_TESTS
    auc_equiv_all = all(a["tost_p"] < ALPHA for a in auc)
    auc_equiv_all_bonf = all(a["tost_p"] < bonf for a in auc)
    rank1_notabove_all = all(r["after_not_above_floor"] for r in rank1)

    out = {"alpha": ALPHA, "bonferroni_threshold": bonf, "n_tests_per_metric": N_TESTS,
           "auc": {"margin": 0.02, "per_channel": auc,
                   "equivalent_all_channels": auc_equiv_all,
                   "equivalent_all_channels_bonferroni": auc_equiv_all_bonf},
           "rank1": {"margin": 0.10, "per_channel": rank1,
                     "after_not_above_floor_all_channels": rank1_notabove_all}}
    with open(config.RESULTS_DIR / "equivalence_tests.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nAUC: after==floor by TOST (margin 0.02) at every channel: "
          f"{auc_equiv_all} (Bonferroni {bonf:.4f}: {auc_equiv_all_bonf})")
    print(f"rank-1: after not above floor at every channel: {rank1_notabove_all}")
    print("-> results/equivalence_tests.json")


if __name__ == "__main__":
    main()
