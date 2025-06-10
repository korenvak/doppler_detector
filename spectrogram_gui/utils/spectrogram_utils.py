import numpy as np
import librosa
from datetime import datetime
import os

import numpy as np
from scipy.signal import spectrogram
from scipy.ndimage import gaussian_filter, median_filter

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
    Sxx_norm = (Sxx_dB - Sxx_dB.min()) / (Sxx_dB.max() - Sxx_dB.min())

    # 4) smooth + median filter
    sigma      = params.get("smooth_sigma", 1.5)
    mf_size    = params.get("median_filter_size", (3, 1))
    Sxx_smoothed = gaussian_filter(Sxx_norm, sigma=sigma)
    Sxx_filt     = median_filter(Sxx_smoothed, size=mf_size)

    return freqs, times, Sxx_norm, Sxx_filt
