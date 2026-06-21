"""Regenerate the paper figures and tables from the result JSON files so every
number is consistent across the paper.

Reads:
    results/sharding_unlearning.json   (S-sweep)
    results/ablations.json             (R-sweep, feature, weak-learner)
    results/reid_auc.json              (leakage-free re-id AUC)
    results/unlearning_speedup.json    (per-subject speedup)
    results/multiseed_eer.json         (multi-seed EER)
    results/unlearning_costs.json      (RQ2 costs)

Run:
    python experiments/13_paper_assets.py

Outputs:
    figures/fig1_rq1_sharding.png, fig2_rq2_unlearning.png,
    figures/fig3_rq3_forgetting.png, fig4_rq4_ablation.png
    paper/tables/*.md, paper/tables/*.tex
"""
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
C = {"sa": "#4c72b0", "uni": "#dd8452", "fs": "#7f7f7f", "g": "#55a868",
     "p": "#8172b3", "r": "#c44e52"}


def load(n):
    return json.load(open(config.RESULTS_DIR / n))


def main() -> None:
    p11 = load("sharding_unlearning.json")
    p11b = load("ablations.json")
    p14 = load("reid_auc.json")
    p15 = load("unlearning_speedup.json")
    p17 = load("multiseed_eer.json")
    p18 = load("unlearning_costs.json")
    F = config.FIGURES_DIR
    tdir = config.PROJECT_ROOT / "paper" / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    sa = {r["S"]: r["open_set_eer"] for r in p11["S_sweep"] if r["mode"] == "subject_aware"}
    un = {r["S"]: r["open_set_eer"] for r in p11["S_sweep"] if r["mode"] == "uniform"}

    # ---- Fig 1: S-sweep ----
    fig, ax = plt.subplots(figsize=(6.5, 4))
    Ss = sorted(sa)
    ax.plot(Ss, [sa[s] for s in Ss], "o-", color=C["sa"], label="subject-aware")
    ax.plot(Ss, [un[s] for s in Ss], "o-", color=C["uni"], label="uniform")
    ax.set_xlabel("number of shards $S$"); ax.set_ylabel("open-set EER (cross-task)")
    ax.set_title("RQ1: utility cost of sharding"); ax.legend()
    fig.tight_layout(); fig.savefig(F / "fig1_rq1_sharding.png"); plt.close(fig)

    # ---- Fig 2: cost + slicing ----
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4))
    a0.bar(["from-scratch", "uniform\nSISA", "subject-aware\nSISA"],
           [p18["from_scratch_cost"], p18["uniform_cost"], p18["subject_aware_median_cost"]],
           color=[C["fs"], C["uni"], C["sa"]])
    a0.set_yscale("log"); a0.set_ylabel("retraining cost (epochs$\\times$samples)")
    a0.set_title("RQ2: cost to forget one subject")
    for i, s in enumerate([p18["from_scratch_shards"], p18["uniform_shards"], 1]):
        v = [p18["from_scratch_cost"], p18["uniform_cost"], p18["subject_aware_median_cost"]][i]
        a0.text(i, v, f"{s} shard(s)", ha="center", va="bottom", fontsize=9)
    Rs = [r["R"] for r in p11b["R_sweep"]]
    a1.plot(Rs, [r["mean_speedup"] for r in p11b["R_sweep"]], "o-", color=C["sa"])
    a1.set_xlabel("number of slices $R$"); a1.set_ylabel("mean unlearning speedup ($\\times$)")
    a1.set_title("RQ2: slicing speedup")
    a2 = a1.twinx()
    a2.plot(Rs, [r["open_set_eer"] for r in p11b["R_sweep"]], "s--", color=C["uni"], alpha=.7)
    a2.set_ylabel("open-set EER", color=C["uni"]); a2.set_ylim(0, 0.06); a2.grid(False)
    fig.tight_layout(); fig.savefig(F / "fig2_rq2_unlearning.png"); plt.close(fig)

    # ---- Fig 3: leakage-free re-id (two panels) ----
    rows = p14["rows"]
    xs = [r["channels"] for r in rows]
    fig, (b0, b1) = plt.subplots(1, 2, figsize=(11, 4.2))
    b0.errorbar(xs, [r["raw_rank1"] for r in rows], yerr=[r.get("raw_rank1_ci", 0) for r in rows],
                marker="o", capsize=3, color=C["p"], label="raw EEG re-id")
    b0.axhline(rows[0]["rank1_chance"], ls="--", c="k", lw=1, label="chance (1/103)")
    b0.set_xscale("log", base=2); b0.set_xticks(xs); b0.set_xticklabels(xs)
    b0.set_xlabel("number of EEG channels")
    b0.set_ylabel("rank-1 identification over 103 subjects")
    b0.set_ylim(0, 0.7); b0.set_title("Re-identifiability persists in the signal"); b0.legend(fontsize=9)
    for key, lab, col in [("raw_auc", "raw features", C["p"]), ("before_auc", "before", C["sa"]),
                          ("after_auc", "after (unlearned)", C["r"]),
                          ("floor_auc", "never-enrolled floor", C["g"])]:
        b1.errorbar(xs, [r[key] for r in rows], yerr=[r[key.replace("_auc", "_std")] for r in rows],
                    marker="o", capsize=3, label=lab, color=col)
    b1.axhline(0.5, ls="--", c="k", lw=1)
    b1.set_xscale("log", base=2); b1.set_xticks(xs); b1.set_xticklabels(xs)
    b1.set_xlabel("number of EEG channels"); b1.set_ylabel("separability AUC (run-disjoint)")
    b1.set_ylim(0.45, 1.02); b1.set_title("After unlearning tracks the never-enrolled floor")
    b1.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(F / "fig3_rq3_forgetting.png"); plt.close(fig)

    # ---- Fig 4: weak-learner + feature ablation ----
    fig, (c0, c1) = plt.subplots(1, 2, figsize=(11, 4))
    pr = p11b["weak_learner_probe"]
    c0.plot([p["windows_per_subject"] for p in pr], [p["open_set_eer"] for p in pr], "o-", color=C["sa"])
    c0.set_xscale("log"); c0.set_xlabel("training windows per subject")
    c0.set_ylabel("open-set EER"); c0.set_title("RQ4: weak-learner curve")
    fa = p11b["feature_ablation"]
    c1.bar([r["features"].split()[0] for r in fa], [r["open_set_eer"] for r in fa], color=C["g"])
    c1.set_ylabel("open-set EER"); c1.set_title("RQ4: feature (band) ablation")
    c1.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(F / "fig4_rq4_ablation.png"); plt.close(fig)

    # ---- Tables ----
    eer = p17["systems"]
    rows_t = [
        ("Monolithic ($S{=}1$)", eer["monolithic"], 1, "1.0"),
        ("Uniform SISA ($S{=}10$)", eer["uniform_S10"], p18["uniform_shards"],
         f"{p18['uniform_speedup']:.1f}"),
        ("Subject-aware SISA ($S{=}10$)", eer["subject_aware_S10"], 1,
         f"{p15['speedup_median']:.0f}"),
    ]
    md = ["| System | open-set EER | shards retrained | unlearning speedup |", "|---|---|---|---|"]
    for n, e, sh, sp in rows_t:
        md.append(f"| {n} | {e['mean_eer']:.4f} $\\pm$ {e['std_eer']:.4f} | {sh} | {sp}$\\times$ |")
    (tdir / "main_results.md").write_text("\n".join(md), encoding="utf-8")
    tex = ["\\begin{tabular}{lccc}", "\\toprule",
           "System & open-set EER & shards retrained & speedup \\\\", "\\midrule"]
    for n, e, sh, sp in rows_t:
        tex.append(f"{n} & {e['mean_eer']:.4f} $\\pm$ {e['std_eer']:.4f} & {sh} & {sp}$\\times$ \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (tdir / "main_results.tex").write_text("\n".join(tex), encoding="utf-8")

    rmd = ["| channels | raw rank-1 ($\\pm$95% CI) | before AUC | after AUC ($\\pm$std) | floor AUC ($\\pm$std) |",
           "|---|---|---|---|---|"]
    for r in rows:
        rmd.append(f"| {r['channels']} | {r['raw_rank1']:.3f} $\\pm$ {r.get('raw_rank1_ci', 0):.3f} | "
                   f"{r['before_auc']:.3f} | {r['after_auc']:.3f} $\\pm$ {r['after_std']:.3f} | "
                   f"{r['floor_auc']:.3f} $\\pm$ {r['floor_std']:.3f} |")
    (tdir / "forgetting.md").write_text("\n".join(rmd), encoding="utf-8")

    smd = ["| S | subject-aware EER | uniform EER |", "|---|---|---|"]
    for S in Ss:
        smd.append(f"| {S} | {sa[S]:.4f} | {un[S]:.4f} |")
    (tdir / "rq1_sweep.md").write_text("\n".join(smd), encoding="utf-8")

    print("regenerated fig1-fig4 and tables.")
    print("\nmain results:")
    print((tdir / "main_results.md").read_text(encoding="utf-8"))
    print("\nspeedup: median {:.1f}x, mean {:.1f}x, range [{:.1f},{:.1f}]".format(
        p15["speedup_median"], p15["speedup_mean"], p15["speedup_min"], p15["speedup_max"]))


if __name__ == "__main__":
    main()
