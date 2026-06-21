"""Run the full replication pipeline in dependency order.

Each step writes its results under ``results/`` and figures under ``figures/``.
The steps are ordered so every input a step reads has already been produced.

Usage:
    python run_all.py            # run every step
    python run_all.py --list     # print the ordered steps and exit
    python run_all.py --from 04  # resume from step 04 (reid-auc) onward

Prerequisite: the curated PhysioNet EEGMMIDB dataset must be reachable (see
README). Step 1 builds the feature cache that every later step consumes.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"

# (label, script) in strict dependency order. Comments note cross-step inputs.
STEPS = [
    ("features",         "01_extract_features.py"),        # builds results/features.npz
    ("sharding",         "02_sharding_and_unlearning.py"), # S-sweep + unlearning cost/exactness
    ("ablations",        "03_ablations.py"),               # R-sweep, feature/weak-learner
    ("reid-auc",         "04_reid_auc.py"),                # run-disjoint separability AUC
    ("speedup",          "05_unlearning_speedup.py"),      # per-subject unlearning speedup
    ("scoring-confound", "06_scoring_confound.py"),        # cosine vs softmax EER control
    ("multiseed-eer",    "07_multiseed_eer.py"),           # main-table EER over 5 seeds
    ("unlearn-costs",    "08_unlearning_costs.py"),        # reads unlearning_speedup.json
    ("exactness",        "09_exactness.py"),               # max parameter difference
    ("reid-rank1",       "10_reid_rank1.py"),              # embedding rank-1 re-id
    ("eer-after-removal", "11_eer_after_removal.py"),      # remaining-subject EER delta
    ("equivalence",      "12_equivalence_tests.py"),       # reads reid_auc + reid_rank1 JSON
    ("paper-assets",     "13_paper_assets.py"),            # figures + tables (reads steps 2-8)
]


def run_step(label: str, script: str) -> None:
    path = EXP / script
    print(f"\n{'=' * 70}\n[{label}] {script}\n{'=' * 70}", flush=True)
    t0 = time.time()
    result = subprocess.run([sys.executable, str(path)], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"\nStep '{label}' ({script}) failed with code {result.returncode}.")
    print(f"[{label}] done in {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the ordered steps and exit")
    ap.add_argument("--from", dest="start", default=None,
                    help="resume from the first step whose script name contains this string")
    args = ap.parse_args()

    if args.list:
        for i, (label, script) in enumerate(STEPS, 1):
            print(f"{i:2d}. {label:18s} {script}")
        return

    steps = STEPS
    if args.start is not None:
        idx = next((i for i, (_, s) in enumerate(STEPS) if args.start in s), None)
        if idx is None:
            sys.exit(f"No step matches --from {args.start!r}.")
        steps = STEPS[idx:]

    t0 = time.time()
    for label, script in steps:
        run_step(label, script)
    print(f"\nAll {len(steps)} step(s) completed in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
