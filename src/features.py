"""Phase 2 - preprocessing and band-power feature extraction.

Pipeline per run (whole-run windowing):
  1. optional zero-phase band-pass filter (SOS, config.BANDPASS)
  2. epoch into fixed windows (config.WINDOW_SAMPLES, step config.WINDOW_STEP)
  3. Welch PSD per window per channel, integrate power in each band
  4. log10(power) -> 320 features per window (64 channels x 5 bands)

Note on normalization: we deliberately store *raw* log band-power here and do
NOT z-score, so no statistics are shared across samples. Per-feature
standardization is applied later, fit on the training split only, to keep the
evaluation leakage-safe.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt, welch

from src import config, data

# --- band-pass filter --------------------------------------------------------
_SOS = None
if config.BANDPASS is not None:
    _lo, _hi = config.BANDPASS
    _nyq = config.SFREQ / 2.0
    _SOS = butter(4, [_lo / _nyq, _hi / _nyq], btype="band", output="sos")


def bandpass(run: np.ndarray) -> np.ndarray:
    """Zero-phase band-pass filter a run [n_samples, n_channels]."""
    if _SOS is None:
        return run
    return sosfiltfilt(_SOS, run, axis=0).astype(np.float32)


# --- windowing ---------------------------------------------------------------
def make_windows(run: np.ndarray) -> np.ndarray:
    """Slice a run into overlapping windows.

    Returns array [n_windows, n_channels, window_samples].
    """
    w = config.WINDOW_SAMPLES
    step = config.WINDOW_STEP
    if run.shape[0] < w:
        return np.empty((0, run.shape[1], w), dtype=run.dtype)
    # sliding_window_view over time -> [n_full, n_channels, w]; subsample by step
    sw = sliding_window_view(run, w, axis=0)[::step]
    return np.ascontiguousarray(sw)


# --- band power --------------------------------------------------------------
def feature_names() -> list[str]:
    return [f"ch{c:02d}_{band}" for c in range(config.N_CHANNELS) for band in config.BANDS]


def band_power(windows: np.ndarray) -> np.ndarray:
    """log10 band power for each window/channel.

    windows : [n_windows, n_channels, window_samples]
    returns : [n_windows, n_channels * n_bands]  (channel-major, band order = config.BANDS)
    """
    if windows.shape[0] == 0:
        return np.empty((0, config.N_FEATURES), dtype=np.float32)
    f, pxx = welch(
        windows, fs=config.SFREQ, nperseg=config.WELCH_NPERSEG,
        noverlap=config.WELCH_NOVERLAP, axis=-1,
    )  # pxx: [n_windows, n_channels, n_freqs]
    feats = []
    for lo, hi in config.BANDS.values():
        mask = (f >= lo) & (f < hi)
        # integrate PSD over the band (trapezoid) -> [n_windows, n_channels]
        bp = np.trapz(pxx[..., mask], f[mask], axis=-1)
        feats.append(bp)
    # stack -> [n_bands, n_windows, n_channels] -> [n_windows, n_channels, n_bands]
    bp = np.stack(feats, axis=0).transpose(1, 2, 0)
    bp = np.log10(bp + config.LOG_OFFSET)
    n_win = bp.shape[0]
    return bp.reshape(n_win, config.N_FEATURES).astype(np.float32)


def extract_run(sid: int, run: int) -> np.ndarray:
    """Full feature pipeline for one run -> [n_windows, 320]."""
    sig = data.load_signal(sid, run)
    sig = bandpass(sig)
    windows = make_windows(sig)
    return band_power(windows)
