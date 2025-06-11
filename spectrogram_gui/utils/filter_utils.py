# File: filter_utils.py

import numpy as np
from scipy.signal import stft, istft
from typing import Optional

def apply_lms(x: np.ndarray, mu: float = 0.01, filter_order: int = 32) -> np.ndarray:
    """Simple LMS adaptive filter returning the error signal."""
    x = x.astype(np.float64, copy=False)
    n = len(x)
    w = np.zeros(filter_order, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    x_pad = np.concatenate([np.zeros(filter_order), x])
    for i in range(n):
        u = x_pad[i : i + filter_order][::-1]
        y_pred = np.dot(w, u)
        e = x[i] - y_pred
        w += 2 * mu * e * u
        y[i] = e
    return y

def apply_nlms(x: np.ndarray, mu: float = 0.01, filter_order: int = 32) -> np.ndarray:
    """
    Normalized LMS adaptive filter returning the error signal.
    """
    x = x.astype(np.float64, copy=False)
    n = len(x)
    w = np.zeros(filter_order, dtype=np.float64)
    eps = 1e-8
    y = np.zeros(n, dtype=np.float64)
    x_padded = np.concatenate([np.zeros(filter_order), x])
    for i in range(n):
        u = x_padded[i : i + filter_order][::-1]
        y_pred = np.dot(w, u)
        e = x[i] - y_pred
        norm = np.dot(u, u) + eps
        w += (mu / norm) * e * u
        y[i] = e
    return y

def apply_ale(
    x: np.ndarray,
    delay: Optional[int] = 1,
    mu: float = 0.01,
    filter_order: int = 32,
) -> np.ndarray:
    """Adaptive Line Enhancer using an LMS filter.

    If ``delay`` is ``None``, the delay is estimated automatically by
    searching for the first minimum of the autocorrelation up to
    ``filter_order`` samples.
    """
    x = x.astype(np.float64, copy=False)
    n = len(x)
    if delay is None:
        ac = np.correlate(x, x, mode="full")[n - 1 : n - 1 + filter_order]
        if len(ac) > 1:
            delay = int(np.argmin(np.abs(ac[1:])) + 1)
        else:
            delay = 1
    w = np.zeros(filter_order, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    x_pad = np.concatenate([np.zeros(delay + filter_order), x])
    for i in range(n):
        u = x_pad[i + delay : i + delay + filter_order][::-1]
        y_pred = np.dot(w, u)
        e = x[i] - y_pred
        w += 2 * mu * e * u
        y[i] = e
    return y

def apply_rls(x: np.ndarray, forgetting_factor: float = 0.99, filter_order: int = 32) -> np.ndarray:
    """Recursive Least Squares adaptive filter returning the error signal."""
    x = x.astype(np.float64, copy=False)
    n = len(x)
    delta = 1.0
    P = np.eye(filter_order, dtype=np.float64) / delta
    w = np.zeros(filter_order, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    x_pad = np.concatenate([np.zeros(filter_order), x])
    eps = 1e-8
    for i in range(n):
        u = x_pad[i : i + filter_order][::-1]
        Pi_u = P.dot(u)
        denom = forgetting_factor + np.dot(u, Pi_u) + eps
        k = Pi_u / denom
        y_pred = np.dot(w, u)
        e = x[i] - y_pred
        w += k * e
        P = (P - np.outer(k, Pi_u)) / forgetting_factor
        y[i] = e
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
