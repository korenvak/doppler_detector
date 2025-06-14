# File: filter_utils.py

import numpy as np
from scipy.signal import stft, istft, butter, sosfilt
from scipy.ndimage import gaussian_filter1d, median_filter
from typing import Optional, List, Union, Tuple

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
    delay: Optional[int] = None,
    mu: float = 0.01,
    filter_order: int = 32,
    test_delays: Optional[Union[int, List[int]]] = None,
    return_all: bool = False,
    return_metrics: bool = False,
    forgetting_factor: Optional[float] = None,
    slope: float = 0.0,
    freq_domain: bool = False,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> Union[np.ndarray, Tuple]:
    """Adaptive Line Enhancer using an LMS filter.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    delay : int or None, optional
        Specific delay to use. If ``None`` (default), the delay that maximizes
        output energy over ``test_delays`` is selected.
    mu : float
        LMS adaptation rate.
    filter_order : int
        Order of the adaptive filter.
    test_delays : int or List[int], optional
        Range of delays to evaluate when ``delay`` is ``None``.  If an integer
        is provided, delays ``1..test_delays`` are evaluated.  Defaults to
        ``1..10``.
    return_all : bool
        If ``True``, return a list of all predictions for each tested delay.
    return_metrics : bool
        If ``True``, also return the SNR metric (in dB) for each tested delay.
    forgetting_factor : float, optional
        Placeholder for backwards compatibility (unused).
    slope : float, optional
        Linear slope factor applied to the delay to better follow
        frequency-drifting tones. Default is ``0.0``.
    freq_domain : bool, optional
        If ``True``, perform the enhancement on STFT magnitudes instead of the
        raw time signal. Default is ``False``.
    n_fft : int, optional
        FFT size when ``freq_domain`` is ``True``. Default is ``1024``.
    hop_length : int, optional
        Hop size when ``freq_domain`` is ``True``. Default is ``512``.

    Returns
    -------
    np.ndarray or tuple
        The predicted signal for the best delay.  If ``return_all`` is ``True``
        the return value is ``(best_y, best_delay, all_y, metrics)`` where
        ``metrics`` is ``None`` when ``return_metrics`` is ``False``.
    """

    def _ale_1d(sig: np.ndarray, d: int) -> np.ndarray:
        """Run ALE on a single 1-D signal for a given delay."""
        n_local = len(sig)
        w = np.zeros(filter_order, dtype=np.float64)
        y_pred = np.zeros(n_local, dtype=np.float64)
        x_pad = np.concatenate([np.zeros(d + filter_order + n_local), sig])
        for i in range(n_local):
            shift = int(round(i * slope))
            u = x_pad[i + d + shift : i + d + shift + filter_order][::-1]
            pred = np.dot(w, u)
            e = sig[i] - pred
            w += 2 * mu * e * u
            y_pred[i] = pred
        return y_pred

    x = x.astype(np.float64, copy=False)

    if freq_domain:
        f, t, Zxx = stft(x, nperseg=n_fft, noverlap=n_fft - hop_length)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        out_mag = np.zeros_like(mag)
        for row in range(mag.shape[0]):
            out_mag[row, :] = apply_ale(
                mag[row, :],
                delay=delay,
                mu=mu,
                filter_order=filter_order,
                test_delays=test_delays,
                slope=slope,
                freq_domain=False,
            )
        Zxx_filt = out_mag * np.exp(1j * phase)
        _, x_out = istft(Zxx_filt, nperseg=n_fft, noverlap=n_fft - hop_length)
        if len(x_out) > len(x):
            x_out = x_out[:len(x)]
        else:
            x_out = np.pad(x_out, (0, len(x) - len(x_out)))
        return x_out

    n = len(x)

    if delay is not None:
        # Single delay case
        best_y = _ale_1d(x, delay)
        metrics = None
        if return_metrics:
            err = x - best_y
            num = np.sum(best_y ** 2)
            den = np.sum(err ** 2) + 1e-8
            metrics = [10 * np.log10(num / den)]
        if return_all:
            return best_y, delay, [best_y], metrics
        return best_y if not return_metrics else (best_y, delay, metrics)

    # Search over multiple delays for best energy
    if test_delays is None:
        delays = list(range(1, 11))
    elif isinstance(test_delays, int):
        delays = list(range(1, test_delays + 1))
    else:
        delays = list(test_delays)

    all_y = []
    metrics = []
    energies = []
    for d in delays:
        yp = _ale_1d(x, d)
        all_y.append(yp)
        energy = float(np.sum(yp ** 2))
        energies.append(energy)
        if return_metrics:
            err = x - yp
            snr = 10 * np.log10(energy / (np.sum(err ** 2) + 1e-8))
            metrics.append(snr)

    best_idx = int(np.argmax(energies))
    best_delay = delays[best_idx]
    best_y = all_y[best_idx]

    if return_all:
        return best_y, best_delay, all_y, metrics if return_metrics else None

    return best_y if not return_metrics else (best_y, best_delay, metrics[best_idx])

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


def apply_gaussian(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply a simple 1D Gaussian smoothing filter."""
    return gaussian_filter1d(x.astype(np.float64, copy=False), sigma)


def apply_median(x: np.ndarray, size: int = 3) -> np.ndarray:
    """Apply a 1D median filter."""
    return median_filter(x.astype(np.float64, copy=False), size=size)


def apply_gabor(x: np.ndarray, freq: float = 0.1, sigma: float = 2.0) -> np.ndarray:
    """Apply a simple 1D Gabor filter."""
    x = x.astype(np.float64, copy=False)
    length = int(6 * sigma)
    if length % 2 == 0:
        length += 1
    half = length // 2
    t = np.arange(-half, half + 1)
    envelope = np.exp(-(t ** 2) / (2 * sigma ** 2))
    carrier = np.cos(2 * np.pi * freq * t)
    kernel = envelope * carrier
    kernel /= np.sum(np.abs(kernel)) + 1e-8
    return np.convolve(x, kernel, mode="same")


def apply_lowpass(x: np.ndarray, cutoff: float, sr: int, order: int = 4) -> np.ndarray:
    """Apply a Butterworth low-pass filter."""
    sos = butter(order, cutoff, btype="lowpass", fs=sr, output="sos")
    return sosfilt(sos, x.astype(np.float64, copy=False))
