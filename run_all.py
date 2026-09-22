from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"

STEPS = [
    ("features",           "01_extract_features.py"),
    ("sharding-sweeps",    "02_sharding_and_sweeps.py"),
    ("paper-figures",      "03_paper_figures.py"),
    ("exactness",          "04_exactness.py"),
    ("eer-after-removal",  "05_eer_after_removal.py"),
    ("erasure",            "06_erasure_by_design.py"),
    ("enrollment-scaling", "07_enrollment_scaling.py"),
    ("session-fusion",     "08_session_fusion.py"),
    ("reid-audit",         "09_reidentification_audit.py"),
    ("equivalence",        "10_equivalence_tests.py"),
]


def run_step(label: str, script: str) -> None:
    print(f"\n{'=' * 70}\n[{label}] {script}\n{'=' * 70}", flush=True)
    t0 = time.time()
    if subprocess.run([sys.executable, str(EXP / script)], cwd=ROOT).returncode != 0:
        sys.exit(f"Step '{label}' ({script}) failed.")
    print(f"[{label}] done in {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
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
