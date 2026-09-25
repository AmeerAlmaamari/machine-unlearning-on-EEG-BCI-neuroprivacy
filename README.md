# Forgetting Is Not Anonymity code

Implementation and experiments for the paper on exact machine unlearning for
EEG-based authentication, on the curated PhysioNet EEG Motor Movement/Imagery
dataset (103 subjects). Running the pipeline regenerates every figure, table, and
statistic reported in the paper.

## Install

Python 3.10.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA GPU is recommended; the pipeline also runs on CPU.

## Dataset

The dataset is not included. Point the code at the CSV/MATLAB release with the
`RLS_DATA_DIR` environment variable:

```bash
export RLS_DATA_DIR=/path/to/EEGMMIDB          # Windows: $env:RLS_DATA_DIR="D:\path\to\EEGMMIDB"
```

`src/data.py` expects `<RLS_DATA_DIR>/CSV files/SUB_001_SIG_01.csv` and the
matching `..._ANN_01.csv` annotation files.

## Run

```bash
python run_all.py            # full pipeline, in order
python run_all.py --list     # list the steps
python run_all.py --from 06  # resume from a step
python experiments/06_erasure_by_design.py   # one step
```

Step 01 builds the feature cache (`results/features.npz`); every later step reads
it. Results are written to `results/` (JSON) and `figures/` (PNG). All figures and
EER numbers are averaged over five seeds (42--46); the exactness, remaining-EER,
and re-identification audits run over all 83 enrolled subjects.

## Steps

| # | Script | Output | Paper artifact |
|---|--------|--------|----------------|
| 01 | `01_extract_features.py`     | `features.npz`, `feature_summary.json` | feature cache (Sec. 5.2) |
| 02 | `02_sharding_and_sweeps.py`  | `sweeps.json` | Table 3, Sec. 6.1, 6.5 |
| 03 | `03_paper_figures.py`        | `fig1_rq1_sharding.png`, `fig2_rq2_unlearning.png`, `fig4_rq4_ablation.png` | Figures 1, 2, 7 |
| 04 | `04_exactness.py`            | `exactness.json` | Sec. 6.2 (max param diff = 0) |
| 05 | `05_eer_after_removal.py`    | `eer_after_removal.json` | Sec. 6.2 (remaining EER) |
| 06 | `06_erasure_by_design.py`    | `erasure_by_design.json`, `cohort_sweep.png` | Table 4, Figure 6 |
| 07 | `07_enrollment_scaling.py`   | `enrollment_scaling.json`, `enrollment_scaling.png` | Figure 3 |
| 08 | `08_session_fusion.py`       | `session_fusion.json`, `session_fusion.png` | Table 5, Figure 4 |
| 09 | `09_reidentification_audit.py` | `reid_audit.json`, `reidentification_audit.png` | Tables 6, 7, Figure 5 |
| 10 | `10_equivalence_tests.py`    | `equivalence.json` | Sec. 6.3 (TOST) |

Step 02 must precede 03, and step 09 must precede 10; the numeric order satisfies
every prerequisite.

## Layout

```
src/            core library (config, data, features, splits, model, sisa, unlearn,
                erasure, metrics, utils)
experiments/    numbered pipeline steps, 01 first
run_all.py      driver
```

## Notes

Seeds are fixed in `src/config.py`. The enrolled/held-out split uses a stream
distinct from the shard assignment, and enrollment (execution runs) is disjoint
from authentication (imagery runs). Subject-aware removal is bit-identical to a
from-scratch retrain under deterministic training; different hardware or library
versions may give numerically different but equivalent results. `results/` and
`figures/` are regenerated and git-ignored.

The PhysioNet EEGMMIDB dataset is governed by its own license; obtain it from
PhysioNet. It is de-identified and labelled by integer subject identifiers.
