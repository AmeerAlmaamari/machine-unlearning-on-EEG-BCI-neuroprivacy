from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "Physionet EEGMMIDB in MATLAB structure and CSV files to leverage accessibility and exploitation"
)
DATA_DIR = Path(os.environ.get("RLS_DATA_DIR", _DEFAULT_DATA_DIR))
DATA_CSV_DIR = DATA_DIR / "CSV files"
DATA_MAT_PATH = DATA_DIR / "MATLAB structure" / "EEGMMIDB_Curated.mat"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
CACHE_DIR = RESULTS_DIR / "cache"
FEATURES_PATH = RESULTS_DIR / "features.npz"

for _d in (RESULTS_DIR, FIGURES_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

N_SUBJECTS = 103
N_RUNS = 12
N_CHANNELS = 64
SFREQ = 160

BANDPASS = (0.5, 40.0)
WINDOW_SEC = 2.0
WINDOW_SAMPLES = int(WINDOW_SEC * SFREQ)
WINDOW_OVERLAP = 0.5
WINDOW_STEP = int(WINDOW_SAMPLES * (1 - WINDOW_OVERLAP))
WINDOW_SOURCE = "whole_run"

WELCH_NPERSEG = 160
WELCH_NOVERLAP = 80
LOG_OFFSET = 1e-12

BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}
N_FEATURES = N_CHANNELS * len(BANDS)

SHARD_COUNTS = [1, 5, 10, 20]
SLICE_COUNTS = [1, 5, 10]
DEFAULT_S = 10
DEFAULT_R = 5

MLP_HIDDEN = [256, 128]
MLP_EPOCHS = 50
MLP_BATCH_SIZE = 128
MLP_LR = 1e-3
