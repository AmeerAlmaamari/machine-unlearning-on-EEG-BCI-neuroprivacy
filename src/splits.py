"""Leakage-safe, run-based train/val/test splitting.

Each subject's 12 runs are partitioned into whole runs: 8 train / 2 val / 2 test.
Because the split is by *run*, no window from a test run can appear in training
(overlapping windows within a run always stay in the same set). The assignment
is a deterministic per-subject permutation seeded by ``config.SEED``.
"""
from __future__ import annotations

import numpy as np

from src import config, data

N_TRAIN, N_VAL, N_TEST = 8, 2, 2

# Cognitive-task structure of the curated runs (verified from annotation codes):
# odd runs  = motor EXECUTION (real movement), task codes {1,2,3} / {7,8,9}
# even runs = motor IMAGERY  (imagined),       task codes {4,5,6} / {10,11,12}
EXECUTION_RUNS = [1, 3, 5, 7, 9, 11]
IMAGERY_RUNS = [2, 4, 6, 8, 10, 12]


def make_task_split(seed: int | None = None) -> dict[int, dict[str, list[int]]]:
    """Leakage-controlled CROSS-TASK split: enrol/validate on motor-execution
    runs, authenticate on motor-imagery runs. Train and test come from different
    cognitive states, so a model cannot pass by memorising within-state artefacts
    (a standard, stronger control for single-session EEG biometrics)."""
    split = {}
    for sid in data.list_subjects():
        split[int(sid)] = {"train": EXECUTION_RUNS[:5], "val": EXECUTION_RUNS[5:],
                           "test": IMAGERY_RUNS}
    return split


def make_run_split(seed: int | None = None) -> dict[int, dict[str, list[int]]]:
    """Return {subject_id: {'train': [...], 'val': [...], 'test': [...]}}."""
    if seed is None:
        seed = config.SEED
    rng = np.random.default_rng(seed)
    split: dict[int, dict[str, list[int]]] = {}
    for sid in data.list_subjects():
        runs = np.array(sorted(data.list_runs(sid)))
        perm = rng.permutation(runs)
        split[int(sid)] = {
            "train": sorted(int(r) for r in perm[:N_TRAIN]),
            "val": sorted(int(r) for r in perm[N_TRAIN:N_TRAIN + N_VAL]),
            "test": sorted(int(r) for r in perm[N_TRAIN + N_VAL:]),
        }
    return split


def holdout_subjects(n_holdout: int = 20, seed: int | None = None):
    """Partition subjects into ENROLLED vs HELD-OUT (never-enrolled). Held-out
    subjects serve as open-set impostors and as the never-trained privacy floor
    for the representation-level forgetting attack."""
    if seed is None:
        seed = config.SEED
    rng = np.random.default_rng(seed + 1)  # distinct stream from sharding
    subs = np.array(data.list_subjects())
    perm = rng.permutation(subs)
    held = sorted(int(s) for s in perm[:n_holdout])
    enrolled = sorted(int(s) for s in perm[n_holdout:])
    return enrolled, held


def split_masks(subject_id: np.ndarray, run_id: np.ndarray,
                split: dict[int, dict[str, list[int]]]):
    """Boolean masks (train, val, test) over feature-table rows."""
    train = np.zeros(len(subject_id), dtype=bool)
    val = np.zeros_like(train)
    test = np.zeros_like(train)
    for sid, parts in split.items():
        m = subject_id == sid
        train |= m & np.isin(run_id, parts["train"])
        val |= m & np.isin(run_id, parts["val"])
        test |= m & np.isin(run_id, parts["test"])
    return train, val, test


def assert_no_leakage(split: dict[int, dict[str, list[int]]]) -> None:
    """Raise if any subject's train/val/test run sets overlap or are incomplete."""
    for sid, parts in split.items():
        tr, va, te = set(parts["train"]), set(parts["val"]), set(parts["test"])
        assert not (tr & va), f"subject {sid}: train/val overlap"
        assert not (tr & te), f"subject {sid}: train/test overlap"
        assert not (va & te), f"subject {sid}: val/test overlap"
        assert len(tr | va | te) == config.N_RUNS, f"subject {sid}: runs missing"
