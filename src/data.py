from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config

ANN_COLUMNS = ["task_code", "dur_s", "dur_samples", "start", "end"]

def subject_str(sid: int) -> str:
    return f"SUB_{sid:03d}"

def sig_path(sid: int, run: int) -> Path:
    return config.DATA_CSV_DIR / f"{subject_str(sid)}_SIG_{run:02d}.csv"

def ann_path(sid: int, run: int) -> Path:
    return config.DATA_CSV_DIR / f"{subject_str(sid)}_ANN_{run:02d}.csv"

def list_subjects() -> list[int]:
    ids = set()
    for p in config.DATA_CSV_DIR.glob("SUB_*_SIG_*.csv"):
        ids.add(int(p.stem.split("_")[1]))
    return sorted(ids)

def list_runs(sid: int) -> list[int]:
    runs = []
    for p in config.DATA_CSV_DIR.glob(f"{subject_str(sid)}_SIG_*.csv"):
        runs.append(int(p.stem.split("_")[3]))
    return sorted(runs)

def load_signal(sid: int, run: int) -> np.ndarray:
    df = pd.read_csv(sig_path(sid, run), header=None, dtype=np.float32)
    return df.to_numpy()

def load_annotation(sid: int, run: int) -> pd.DataFrame:
    return pd.read_csv(ann_path(sid, run), header=None, names=ANN_COLUMNS)
