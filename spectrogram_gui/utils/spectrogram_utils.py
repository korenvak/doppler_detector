import numpy as np
import librosa
from datetime import datetime
import os
import re

from scipy.signal import spectrogram
from scipy.ndimage import gaussian_filter, median_filter


def parse_timestamp_from_filename(path: str):
    """Return a ``datetime`` parsed from ``path`` or ``None``."""
    patterns = [
        r"(\d{4}-\d{2}-\d{2})[ _](\d{2})-(\d{2})-(\d{2})",
    ]
    text = os.path.normpath(path)
    for pat in patterns:
        m = re.search(pat, text)
        if not m:
            continue
        date_part = m.group(1)
        time_part = ":".join(m.group(i) for i in range(2, 5))
        try:
            return datetime.strptime(
                f"{date_part} {time_part}", "%Y-%m-%d %H:%M:%S"
            )
        except ValueError:
            pass
    return None

def compute_spectrogram(y, sr, filepath, params):
    """
    1) Compute magnitude spectrogram with Blackman-Harris window.
    2) Convert to decibels.
    3) Normalize to [0,1].
    4) Apply Gaussian smoothing then median filtering.
    Returns: freqs, times, Sxx_norm, Sxx_filt
    """
    # 1) window and overlap
    nperseg  = params["window_size"]
    noverlap = int(params["overlap"] / 100 * nperseg)

    # raw spectrogram
    freqs, times, Sxx = spectrogram(
        y,
        fs=sr,
        nperseg=nperseg,
        noverlap=noverlap,
        window="blackmanharris",
        scaling="density",
        mode="magnitude"
    )

    # 2) convert to dB
    Sxx_dB = 10.0 * np.log10(Sxx + 1e-10)

    # 3) normalize
    range_db = Sxx_dB.max() - Sxx_dB.min()
    if range_db <= 0:
        Sxx_norm = np.zeros_like(Sxx_dB)
    else:
        Sxx_norm = (Sxx_dB - Sxx_dB.min()) / range_db
    Sxx_norm = np.clip(Sxx_norm, 0.0, 1.0)

    # 4) smooth + median filter
    sigma      = params.get("smooth_sigma", 1.5)
    mf_size    = params.get("median_filter_size", (3, 1))
    Sxx_smoothed = gaussian_filter(Sxx_norm, sigma=sigma)
    Sxx_filt     = median_filter(Sxx_smoothed, size=mf_size)

    return freqs, times, Sxx_norm, Sxx_filt
