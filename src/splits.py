from __future__ import annotations

import numpy as np

from src import config, data

N_TRAIN, N_VAL, N_TEST = 8, 2, 2

EXECUTION_RUNS = [1, 3, 5, 7, 9, 11]
IMAGERY_RUNS = [2, 4, 6, 8, 10, 12]

def make_task_split(seed: int | None = None) -> dict[int, dict[str, list[int]]]:
    split = {}
    for sid in data.list_subjects():
        split[int(sid)] = {"train": EXECUTION_RUNS[:5], "val": EXECUTION_RUNS[5:],
                           "test": IMAGERY_RUNS}
    return split

def make_run_split(seed: int | None = None) -> dict[int, dict[str, list[int]]]:
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
    if seed is None:
        seed = config.SEED
    rng = np.random.default_rng(seed + 1)
    subs = np.array(data.list_subjects())
    perm = rng.permutation(subs)
    held = sorted(int(s) for s in perm[:n_holdout])
    enrolled = sorted(int(s) for s in perm[n_holdout:])
    return enrolled, held

def split_masks(subject_id: np.ndarray, run_id: np.ndarray,
                split: dict[int, dict[str, list[int]]]):
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
    for sid, parts in split.items():
        tr, va, te = set(parts["train"]), set(parts["val"]), set(parts["test"])
        assert not (tr & va), f"subject {sid}: train/val overlap"
        assert not (tr & te), f"subject {sid}: train/test overlap"
        assert not (va & te), f"subject {sid}: val/test overlap"
        assert len(tr | va | te) == config.N_RUNS, f"subject {sid}: runs missing"
