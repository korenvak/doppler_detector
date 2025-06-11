# File: filter_utils.py

import numpy as np
from scipy.signal import stft, istft
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
    delay: Optional[int] = 1,
    mu: float = 0.01,
    filter_order: int = 32,
    test_delays: Optional[Union[int, List[int]]] = None,
    return_all: bool = False,
    return_metrics: bool = False,
    forgetting_factor: Optional[float] = None,
) -> Union[np.ndarray, Tuple]:
    """Adaptive Line Enhancer using an LMS filter.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    delay : int or None, optional
        Specific delay to use.  If ``None``, the delay that maximizes output
        energy over ``test_delays`` is selected.
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
        If ``True``, also return the energy based metric for each delay.
    forgetting_factor : float, optional
        Placeholder for backwards compatibility (unused).

    Returns
    -------
    np.ndarray or tuple
        The predicted signal for the best delay.  If ``return_all`` is ``True``
        the return value is ``(best_y, best_delay, all_y, metrics)`` where
        ``metrics`` is ``None`` when ``return_metrics`` is ``False``.
    """

    def _run_ale(d: int) -> np.ndarray:
        """Run ALE for a single delay and return the predicted signal."""
        w = np.zeros(filter_order, dtype=np.float64)
        y_pred = np.zeros(n, dtype=np.float64)
        x_pad = np.concatenate([np.zeros(d + filter_order), x])
        for i in range(n):
            u = x_pad[i + d : i + d + filter_order][::-1]
            pred = np.dot(w, u)
            e = x[i] - pred
            w += 2 * mu * e * u
            y_pred[i] = pred
        return y_pred

    x = x.astype(np.float64, copy=False)
    n = len(x)

    if delay is not None:
        # Single delay case
        best_y = _run_ale(delay)
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
        yp = _run_ale(d)
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
