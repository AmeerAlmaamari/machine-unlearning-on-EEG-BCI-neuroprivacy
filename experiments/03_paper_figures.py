from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 120, "axes.axisbelow": True})
C = {"sa": "#4c72b0", "uni": "#dd8452", "fs": "#7f7f7f", "g": "#55a868"}


def main() -> None:
    d = json.load(open(config.RESULTS_DIR / "sweeps.json"))
    F = config.FIGURES_DIR

    sa = {x["S"]: x for x in d["S_sweep"] if x["mode"] == "subject_aware"}
    un = {x["S"]: x for x in d["S_sweep"] if x["mode"] == "uniform"}
    Ss = sorted(sa)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(Ss, [sa[s]["eer_mean"] for s in Ss], yerr=[sa[s]["eer_std"] for s in Ss],
                marker="o", capsize=4, color=C["sa"], label="subject-aware")
    ax.errorbar(Ss, [un[s]["eer_mean"] for s in Ss], yerr=[un[s]["eer_std"] for s in Ss],
                marker="s", capsize=4, color=C["uni"], label="uniform")
    ax.set_xlabel("number of shards $S$")
    ax.set_ylabel("open-set EER (cross-task)")
    ax.set_title("Utility cost of sharding (5 seeds)")
    ax.legend()
    fig.tight_layout(); fig.savefig(F / "fig1_rq1_sharding.png"); plt.close(fig)

    c = d["cost"]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4))
    names = ["from-scratch", "uniform\nSISA", "subject-aware\nSISA"]
    vals = [c["from_scratch_mean"], c["uniform_mean"], c["subject_aware_median_mean"]]
    errs = [c["from_scratch_std"], c["uniform_std"], c["subject_aware_median_std"]]
    a0.bar(names, vals, yerr=errs, capsize=4, color=[C["fs"], C["uni"], C["sa"]])
    a0.set_yscale("log")
    a0.set_ylabel("retraining cost (epochs $\\times$ samples)")
    a0.set_title("Cost to forget one subject")
    for i, s in enumerate([10, 10, 1]):
        a0.text(i, vals[i], f"{s} shard(s)", ha="center", va="bottom", fontsize=9)

    Rs = [x["R"] for x in d["R_sweep"]]
    a1.errorbar(Rs, [x["speedup_mean"] for x in d["R_sweep"]],
                yerr=[x["speedup_std"] for x in d["R_sweep"]],
                marker="o", capsize=4, color=C["sa"])
    a1.set_xlabel("number of slices $R$")
    a1.set_ylabel("mean unlearning speedup ($\\times$)")
    a1.set_title("Slicing speedup (5 seeds)")
    a2 = a1.twinx()
    a2.errorbar(Rs, [x["eer_mean"] for x in d["R_sweep"]],
                yerr=[x["eer_std"] for x in d["R_sweep"]],
                marker="s", ls="--", capsize=4, color=C["uni"], alpha=0.8)
    a2.set_ylabel("open-set EER", color=C["uni"])
    a2.set_ylim(0, 0.06); a2.grid(False)
    fig.tight_layout(); fig.savefig(F / "fig2_rq2_unlearning.png"); plt.close(fig)

    fig, (b0, b1) = plt.subplots(1, 2, figsize=(11, 4))
    pr = d["weak_learner_probe"]
    b0.errorbar([p["windows_per_subject"] for p in pr], [p["eer_mean"] for p in pr],
                yerr=[p["eer_std"] for p in pr], marker="o", capsize=4, color=C["sa"])
    b0.set_xscale("log")
    b0.set_xlabel("training windows per subject")
    b0.set_ylabel("open-set EER")
    b0.set_title("Weak-learner curve (5 seeds)")

    fa = d["feature_ablation"]
    labels = [x["features"].split()[0] for x in fa]
    b1.bar(labels, [x["eer_mean"] for x in fa], yerr=[x["eer_std"] for x in fa],
           capsize=4, color=C["g"])
    b1.set_ylabel("open-set EER")
    b1.set_title("Feature (band) ablation (5 seeds)")
    b1.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(F / "fig4_rq4_ablation.png"); plt.close(fig)


if __name__ == "__main__":
    main()
