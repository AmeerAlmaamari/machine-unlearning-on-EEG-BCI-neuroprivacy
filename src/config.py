"""Central configuration for the SISA / EEG-unlearning replication package.

All paths, hyperparameters, and dataset constants live here so the experiments
are config-driven and reproducible. Import with:

    from src import config

The dataset directory can be overridden with the ``RLS_DATA_DIR`` environment
variable; otherwise it defaults to a folder next to this package (see README).
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The curated PhysioNet EEGMMIDB dataset (CSV + MATLAB). Not redistributed with
# this package; place it under the default path below or point RLS_DATA_DIR at it.
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

# --- Reproducibility ---------------------------------------------------------
SEED = 42

# --- Dataset constants -------------------------------------------------------
N_SUBJECTS = 103          # SUB_001..SUB_103 after curation
N_RUNS = 12               # task runs per subject
N_CHANNELS = 64           # columns in each *_SIG_*.csv (no header)
SFREQ = 160               # Hz, sampling rate of EEGMMIDB

# --- Preprocessing / features ------------------------------------------------
BANDPASS = (0.5, 40.0)    # Hz; set to None to disable
WINDOW_SEC = 2.0          # window length
WINDOW_SAMPLES = int(WINDOW_SEC * SFREQ)   # 320
WINDOW_OVERLAP = 0.5      # fractional overlap between consecutive windows
WINDOW_STEP = int(WINDOW_SAMPLES * (1 - WINDOW_OVERLAP))  # 160
WINDOW_SOURCE = "whole_run"   # use the entire run (ignore task codes)

# Welch PSD parameters for band power (1 s segments -> 1 Hz resolution).
WELCH_NPERSEG = 160
WELCH_NOVERLAP = 80
LOG_OFFSET = 1e-12        # added before log10 of band power

# Frequency bands for band-power features (Hz). 64 channels x 5 bands = 320 feats.
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}
N_FEATURES = N_CHANNELS * len(BANDS)       # 320

# --- SISA knobs --------------------------------------------------------------
SHARD_COUNTS = [1, 5, 10, 20]   # S sweep
SLICE_COUNTS = [1, 5, 10]       # R sweep
DEFAULT_S = 10
DEFAULT_R = 5

# --- Model -------------------------------------------------------------------
MLP_HIDDEN = [256, 128]
MLP_EPOCHS = 50
MLP_BATCH_SIZE = 128
MLP_LR = 1e-3
