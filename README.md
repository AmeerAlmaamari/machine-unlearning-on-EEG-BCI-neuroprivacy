# Replication package

**Forgetting Is Not Anonymity: Exact Machine Unlearning for EEG-Based Authentication**

This package reproduces every figure, table, and statistical test in the paper
from the curated PhysioNet EEG Motor Movement/Imagery dataset. It contains the
core library (`src/`), the experiment scripts (`experiments/`), and a driver
(`run_all.py`) that runs them in dependency order.

---

## Quick start

```bash
# 1. install (Python 3.10)
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. make the dataset reachable (see Section 3)
export RLS_DATA_DIR=/path/to/EEGMMIDB                  # Windows PS: $env:RLS_DATA_DIR="D:\path\to\EEGMMIDB"

# 3. run the whole pipeline (builds the feature cache, then every experiment,
#    then the figures and tables)
python run_all.py
```

When it finishes, the regenerated figures are in `figures/`, the numeric results
in `results/`, and the paper tables in `paper/tables/`. Section 4 explains each
command; Section 5 maps every script to the paper figure/table it produces.

---

## 1. Layout

```
replication/
  README.md
  requirements.txt
  run_all.py                 driver: runs all steps in order
  src/                       core library (imported as `from src import ...`)
    config.py                paths, seed, dataset constants, hyperparameters
    data.py                  CSV dataset access
    features.py              band-pass + windowing + Welch band-power features
    splits.py                cross-task and held-out (open-set) splitting
    model.py                 MLP constituent, training, embeddings
    sisa.py                  sharding, slicing, checkpointing, aggregation
    unlearn.py               subject-level exact removal and cost accounting
    metrics.py               EER, TAR@FAR, verification report
    utils.py                 seeding and device selection
  experiments/               one script per pipeline step (see Section 5)
```

Running the pipeline creates three output directories inside this folder:
`results/` (JSON results and the `features.npz` cache), `figures/` (PNGs), and
`paper/tables/` (Markdown and LaTeX tables). All three are regenerated from
scratch and are git-ignored.

---

## 2. Requirements

- Python 3.10
- The packages in `requirements.txt` (NumPy, pandas, SciPy, scikit-learn,
  matplotlib, pyarrow, PyTorch 2.5.1)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA 12.1 GPU is recommended (the paper used a single NVIDIA RTX 3070 Ti). The
pipeline also runs on CPU; see the determinism note in Section 6.

---

## 3. Dataset

The curated PhysioNet EEGMMIDB dataset is not redistributed here. It is the
CSV + MATLAB release with 103 subjects, 12 task runs each, 64 channels at 160 Hz.
The CSV layout expected by `src/data.py` is:

```
<DATA_DIR>/CSV files/SUB_001_SIG_01.csv   EEG signal, no header, 64 columns
<DATA_DIR>/CSV files/SUB_001_ANN_01.csv   annotations, no header, 5 columns
...
```

Point the code at the dataset in either of two ways:

- set the environment variable `RLS_DATA_DIR` to the dataset folder (recommended):

  ```bash
  # Windows (PowerShell):  $env:RLS_DATA_DIR = "D:\path\to\EEGMMIDB"
  # Linux/macOS:           export RLS_DATA_DIR=/path/to/EEGMMIDB
  ```

- or place the dataset folder next to this package under the default name
  `Physionet EEGMMIDB in MATLAB structure and CSV files to leverage accessibility and exploitation/`.

To confirm the path is correct before a long run, extract features alone first
(`python experiments/01_extract_features.py`); it fails immediately with a clear
file-not-found error if the dataset is not reachable.

---

## 4. Running the experiment

### Run everything

```bash
python run_all.py
```

This executes all 13 steps in the order shown in Section 5. Step 01 builds the
feature cache `results/features.npz`; every later step reads it. The driver
stops with a non-zero exit code if any step fails, naming the step.

### Run, list, or resume specific steps

```bash
python run_all.py --list            # print the ordered steps and exit
python run_all.py --from 04         # resume from step 04 (reid-auc) onward
python experiments/07_multiseed_eer.py   # run a single step directly
```

A single step can be run on its own once its inputs exist. The only hard
prerequisites are: step 01 before everything; step 05 before step 08; steps 04
and 10 before step 12; and steps 02–08 before step 13. Following the numeric
order always satisfies these.

### Cost and expected duration

Every step trains many small MLPs. Total wall-clock time is hardware dependent;
plan for roughly one to three hours end-to-end on a single modern GPU, and
several times longer on CPU. The heaviest steps are the channel-sweep re-id
audits (04 and 10) and the multi-seed EER (07); step 01 is I/O-bound feature
extraction. Each step prints progress as it runs.

### Verifying the outputs

After a full run, `results/` should contain these twelve JSON files plus
`features.npz`:

```
feature_summary.json        sharding_unlearning.json   ablations.json
reid_auc.json               unlearning_speedup.json    scoring_confound.json
multiseed_eer.json          unlearning_costs.json      exactness.json
reid_rank1.json             eer_after_removal.json     equivalence_tests.json
```

`figures/` should contain `fig1_rq1_sharding.png`, `fig2_rq2_unlearning.png`,
`fig3_rq3_forgetting.png`, `fig4_rq4_ablation.png`, `reid_auc_channels.png`, and
`reid_rank1.png`. `paper/tables/` should contain `main_results.md`,
`main_results.tex`, `forgetting.md`, and `rq1_sweep.md`. The final step
(`13_paper_assets.py`) prints the main results table and the speedup summary so
the headline numbers can be read off the console directly.

---

## 5. Steps and paper artifacts

The scripts are numbered in run order; the table below gives what each one reads
and produces. Inputs other than the shared feature cache are noted.

| # | Script | Reads (besides `features.npz`) | Produces | Paper artifact |
|---|--------|-------------------------------|----------|----------------|
| 01 | `01_extract_features.py` | dataset | `features.npz`, `feature_summary.json` | feature cache (Sec. 5.2) |
| 02 | `02_sharding_and_unlearning.py` | | `sharding_unlearning.json` | Figure 1 (S-sweep) |
| 03 | `03_ablations.py` | | `ablations.json` | Figure 2 (right), Figure 4 |
| 04 | `04_reid_auc.py` | | `reid_auc.json` | Table 4, Figure 3 |
| 05 | `05_unlearning_speedup.py` | | `unlearning_speedup.json` | Table 2 (speedup) |
| 06 | `06_scoring_confound.py` | | `scoring_confound.json` | Sec. 6.1 (scorer control) |
| 07 | `07_multiseed_eer.py` | | `multiseed_eer.json` | Table 2 (EER) |
| 08 | `08_unlearning_costs.py` | `unlearning_speedup.json` | `unlearning_costs.json` | Figure 2 (left) |
| 09 | `09_exactness.py` | | `exactness.json` | Sec. 6.2 (max param diff = 0) |
| 10 | `10_reid_rank1.py` | | `reid_rank1.json`, `figures/reid_rank1.png` | Table 5, Figure 5 |
| 11 | `11_eer_after_removal.py` | | `eer_after_removal.json` | Sec. 6.2 (EER after removal) |
| 12 | `12_equivalence_tests.py` | `reid_auc.json`, `reid_rank1.json` | `equivalence_tests.json` | Sec. 6.3 (TOST, non-superiority) |
| 13 | `13_paper_assets.py` | steps 02, 03, 04, 05, 07, 08 | `figures/fig1..fig4`, `paper/tables/*` | Figures 1–4, Tables 2/4 |

---

## 6. Reproducibility notes

- **Seeds.** All randomness is seeded from `config.SEED` (42). The EER in Table 2
  is averaged over seeds 42–46 inside `07_multiseed_eer.py`. The held-out subject
  partition uses a distinct stream (`seed + 1`) from the shard assignment, so the
  open-set impostor pool cannot leak into sharding.
- **Leakage control.** Enrollment uses motor-execution runs and authentication
  uses motor-imagery runs (`splits.make_task_split`). The re-identification audit
  uses a run-disjoint gallery/probe split so overlapping windows never cross the
  train/test boundary.
- **Exactness and determinism.** Subject-aware removal is exact by construction:
  training resumes from the checkpoint taken before the removed subject's slice,
  and with a fixed seed this reproduces a from-scratch retrain of that shard.
  `09_exactness.py` verifies this as a maximum absolute parameter difference.
  Bit-identical reproduction assumes deterministic training; different hardware,
  CUDA, or library versions can yield numerically different but statistically
  equivalent results.
- **Configuration.** Every path, seed, and hyperparameter is in `src/config.py`;
  no value is hard-coded elsewhere. Changing the dataset path is the only setup
  step (Section 3).
- **Outputs.** `results/`, `figures/`, and `paper/tables/` are regenerated by the
  pipeline and are git-ignored; they are not shipped with this package.

---

## 7. Data terms

The code in this package is released for research replication. The PhysioNet
EEGMMIDB dataset is governed by its own license and terms of use; obtain it from
PhysioNet and comply with those terms. The data is de-identified and labelled by
integer subject identifiers only.
