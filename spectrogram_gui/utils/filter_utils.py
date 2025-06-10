# File: filter_utils.py

import numpy as np
from scipy.signal import stft, istft

def apply_nlms(x: np.ndarray, mu: float = 0.01, filter_order: int = 32) -> np.ndarray:
    """
    Normalized LMS adaptive filter.
    x: input signal segment
    mu: step size (0 < mu < 1)
    filter_order: number of taps
    Returns predicted component (enhanced output).
    """
    n = len(x)
    w = np.zeros(filter_order, dtype=float)
    eps = 1e-8
    y = np.zeros(n, dtype=float)
    x_padded = np.concatenate([np.zeros(filter_order), x])
    for i in range(n):
        u = x_padded[i + filter_order - 1 : i - 1 : -1]
        y_pred = np.dot(w, u)
        e = x[i] - y_pred
        norm = np.dot(u, u) + eps
        w += (mu / norm) * e * u
        y[i] = y_pred
    return y

def apply_ale(x: np.ndarray, delay: int = 1, forgetting_factor: float = 0.995, filter_order: int = 32) -> np.ndarray:
    """
    Adaptive Line Enhancer (ALE) via RLS.
    x: input signal segment
    delay: samples delay for reference
    forgetting_factor: RLS lambda (0 < λ < 1)
    filter_order: number of taps
    Returns predicted component (enhanced output).
    """
    n = len(x)
    delta = 0.01
    P = np.eye(filter_order) / delta
    w = np.zeros(filter_order, dtype=float)
    y = np.zeros(n, dtype=float)
    x_delayed = np.concatenate([np.zeros(delay), x])[:n]
    x_pad = np.concatenate([np.zeros(filter_order), x_delayed])
    for i in range(n):
        u = x_pad[i + filter_order - 1 : i - 1 : -1]
        y_pred = np.dot(w, u)
        e = x[i] - y_pred
        Pi_u = P.dot(u)
        k = Pi_u / (forgetting_factor + u.dot(Pi_u))
        w += k * e
        P = (P - np.outer(k, Pi_u)) / forgetting_factor
        y[i] = y_pred
    return y

def apply_wiener(x: np.ndarray, noise_db: float = -20, window_size: int = 1024, overlap: int = 512) -> np.ndarray:
    """
    Wiener filter via spectral subtraction.
    x: input signal segment
    noise_db: noise estimate in dBFS
    window_size, overlap: STFT parameters
    Returns enhanced time-domain signal.
    """
    noise_pow = 10 ** (noise_db / 10)
    f, t, Zxx = stft(x, nperseg=window_size, noverlap=overlap)
    Sxx = np.abs(Zxx) ** 2
    G = Sxx / (Sxx + noise_pow)
    Zxx_w = Zxx * G
    _, x_wiener = istft(Zxx_w, nperseg=window_size, noverlap=overlap)
    if len(x_wiener) > len(x):
        return x_wiener[:len(x)]
    else:
        return np.pad(x_wiener, (0, len(x) - len(x_wiener)))
