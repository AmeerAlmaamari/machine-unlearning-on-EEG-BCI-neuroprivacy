from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt, welch

from src import config, data

_SOS = None
if config.BANDPASS is not None:
    _lo, _hi = config.BANDPASS
    _nyq = config.SFREQ / 2.0
    _SOS = butter(4, [_lo / _nyq, _hi / _nyq], btype="band", output="sos")

def bandpass(run: np.ndarray) -> np.ndarray:
    if _SOS is None:
        return run
    return sosfiltfilt(_SOS, run, axis=0).astype(np.float32)

def make_windows(run: np.ndarray) -> np.ndarray:
    w = config.WINDOW_SAMPLES
    step = config.WINDOW_STEP
    if run.shape[0] < w:
        return np.empty((0, run.shape[1], w), dtype=run.dtype)

    sw = sliding_window_view(run, w, axis=0)[::step]
    return np.ascontiguousarray(sw)

def feature_names() -> list[str]:
    return [f"ch{c:02d}_{band}" for c in range(config.N_CHANNELS) for band in config.BANDS]

def band_power(windows: np.ndarray) -> np.ndarray:
    if windows.shape[0] == 0:
        return np.empty((0, config.N_FEATURES), dtype=np.float32)
    f, pxx = welch(
        windows, fs=config.SFREQ, nperseg=config.WELCH_NPERSEG,
        noverlap=config.WELCH_NOVERLAP, axis=-1,
    )
    feats = []
    for lo, hi in config.BANDS.values():
        mask = (f >= lo) & (f < hi)

        bp = np.trapz(pxx[..., mask], f[mask], axis=-1)
        feats.append(bp)

    bp = np.stack(feats, axis=0).transpose(1, 2, 0)
    bp = np.log10(bp + config.LOG_OFFSET)
    n_win = bp.shape[0]
    return bp.reshape(n_win, config.N_FEATURES).astype(np.float32)

def extract_run(sid: int, run: int) -> np.ndarray:
    sig = data.load_signal(sid, run)
    sig = bandpass(sig)
    windows = make_windows(sig)
    return band_power(windows)
