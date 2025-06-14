# File: filter_utils.py

from __future__ import annotations

import numpy as np
import time
from scipy.signal import stft, istft, butter, sosfilt
from scipy.ndimage import gaussian_filter1d, median_filter, label
from typing import Optional, List, Union, Tuple

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - numba may not be installed
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        def wrapper(func):
            return func

        return wrapper


@njit(nopython=True)
def _ale_core(x: np.ndarray, delay: int, mu: float, order: int, slope: float) -> np.ndarray:
    """Return the predicted component of ``x`` using an LMS adaptive filter."""
    n = len(x)
    w = np.zeros(order, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    pad = np.concatenate((np.zeros(delay + order + n), x))
    for i in range(n):
        shift = int(np.rint(i * slope))
        start = i + delay + shift
        u = pad[start : start + order][::-1]
        pred = 0.0
        for j in range(order):
            pred += w[j] * u[j]
        err = x[i] - pred
        for j in range(order):
            w[j] += 2 * mu * err * u[j]
        y[i] = pred
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
    delay: Optional[int] = 3,
    mu: float = 0.01,
    filter_order: int = 32,
    test_delays: Optional[Union[int, List[int]]] = None,
    return_all: bool = False,
    return_metrics: bool = False,
    slope: float = 0.0,
    freq_domain: bool = False,
    return_error: bool = False,
    n_fft: int = 1024,
    hop_length: int = 512,
    fast_mode: bool = False,
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
        is provided, delays ``1..test_delays`` are evaluated. Defaults to
        ``2..4`` for faster operation.
    return_all : bool
        If ``True``, return a list of all predictions for each tested delay.
    return_metrics : bool
        If ``True``, also return the SNR metric (in dB) for each tested delay.
    slope : float, optional
        Linear slope factor applied to the delay to better follow
        frequency-drifting tones. Default is ``0.0``.
    freq_domain : bool, optional
        If ``True``, perform the enhancement on STFT magnitudes instead of the
        raw time signal. Default is ``False``.
    return_error : bool, optional
        If ``True``, return the prediction error instead of the enhanced
        component. Default is ``False``.
    n_fft : int, optional
        FFT size when ``freq_domain`` is ``True``. Default is ``1024``.
    hop_length : int, optional
        Hop size when ``freq_domain`` is ``True``. Default is ``512``.
    fast_mode : bool, optional
        If ``True``, skip delay search and use ``delay`` or 3 by default.

    Returns
    -------
    np.ndarray or tuple
        The predicted signal for the best delay unless ``return_error`` is
        ``True`` in which case the prediction error is returned. If
        ``return_all`` is ``True`` the return value is
        ``(best_y, best_delay, all_y, metrics)`` where ``metrics`` is ``None``
        when ``return_metrics`` is ``False``.
    """

    def _ale_1d(sig: np.ndarray, d: int) -> np.ndarray:
        return _ale_core(sig, d, mu, filter_order, slope)

    x = x.astype(np.float64, copy=False)

    if len(x) > 500000:
        print("[ALE] Input too long, skipping enhancement")
        return x

    start_t = time.perf_counter()

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
                fast_mode=fast_mode,
                return_error=return_error,
            )
        Zxx_filt = out_mag * np.exp(1j * phase)
        _, x_out = istft(Zxx_filt, nperseg=n_fft, noverlap=n_fft - hop_length)
        if len(x_out) > len(x):
            x_out = x_out[:len(x)]
        else:
            x_out = np.pad(x_out, (0, len(x) - len(x_out)))
        # normalize amplitude to input RMS to avoid vanishing output
        in_rms = np.sqrt(np.mean(x ** 2))
        out_rms = np.sqrt(np.mean(x_out ** 2)) + 1e-8
        if out_rms > 0:
            x_out *= in_rms / out_rms
        print(f"[ALE] freq-domain {time.perf_counter()-start_t:.2f}s")
        return x_out

    n = len(x)

    if fast_mode:
        d = delay if delay is not None else 3
        best_y = _ale_1d(x, d)
        print(f"[ALE] fast mode delay={d} took {time.perf_counter()-start_t:.2f}s")
        return x - best_y if return_error else best_y

    if delay is not None:
        # Single delay case
        best_y = _ale_1d(x, delay)
        metrics = None
        if return_metrics:
            err = x - best_y
            num = np.sum(best_y ** 2)
            den = np.sum(err ** 2) + 1e-8
            metrics = [10 * np.log10(num / den)]
        result = x - best_y if return_error else best_y
        if return_all:
            return result, delay, [result], metrics
        return result if not return_metrics else (result, delay, metrics)

    # Search over multiple delays for best energy
    if test_delays is None:
        delays = list(range(2, 5))
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
        result_list = [x - y if return_error else y for y in all_y]
        return (
            result_list[best_idx],
            best_delay,
            result_list,
            metrics if return_metrics else None,
        )

    print(f"[ALE] processed in {time.perf_counter()-start_t:.2f}s")
    result = x - best_y if return_error else best_y
    return result if not return_metrics else (result, best_delay, metrics[best_idx])




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




def apply_wiener_adaptive(x: np.ndarray, window_size: int = 1024) -> np.ndarray:
    """Wiener filter with per-frequency noise estimation."""
    f, t, Zxx = stft(x, nperseg=window_size)
    Sxx = np.abs(Zxx) ** 2
    noise = np.zeros_like(Sxx)
    for i in range(Sxx.shape[0]):
        vals = np.sort(Sxx[i])
        noise[i] = np.median(vals[: max(1, len(vals)//4)])
    gain = np.maximum(Sxx / (Sxx + noise), 0.1)
    Zxx_f = Zxx * gain
    _, out = istft(Zxx_f, nperseg=window_size)
    if len(out) > len(x):
        out = out[: len(x)]
    else:
        out = np.pad(out, (0, len(x) - len(out)))
    return out


def apply_tv_denoising_doppler(
    Sxx: np.ndarray,
    weight_freq: float = 0.01,
    preserve_edges: bool = True,
) -> np.ndarray:
    """TV denoising that preserves Doppler tracks."""
    from skimage.restoration import denoise_tv_chambolle
    from skimage.filters import sobel

    if preserve_edges:
        edges = sobel(np.log1p(Sxx))
        edge_mask = edges > np.percentile(edges, 90)
        adaptive = np.where(edge_mask, weight_freq * 0.1, weight_freq)
        out = np.zeros_like(Sxx)
        for i in range(Sxx.shape[0]):
            out[i] = denoise_tv_chambolle(
                Sxx[i], weight=float(adaptive[i].mean())
            )
        return out
    return denoise_tv_chambolle(Sxx, weight=weight_freq, channel_axis=None)


def apply_tv_denoising_doppler_wave(
    x: np.ndarray,
    weight: float = 0.1,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    """Apply Doppler-preserving TV denoising in the STFT domain."""
    f, t, Zxx = stft(x, nperseg=n_fft, noverlap=n_fft - hop_length)
    mag = np.abs(Zxx) ** 2
    phase = np.angle(Zxx)
    mag_f = apply_tv_denoising_doppler(mag, weight_freq=weight)
    Zxx_f = np.sqrt(mag_f) * np.exp(1j * phase)
    _, out = istft(Zxx_f, nperseg=n_fft, noverlap=n_fft - hop_length)
    if len(out) > len(x):
        out = out[: len(x)]
    else:
        out = np.pad(out, (0, len(x) - len(out)))
    return out


def apply_ale_2d_doppler(
    Sxx: np.ndarray,
    track_width: int = 5,
    delay: int = 3,
    mu: float = 0.01,
    filter_order: int = 32,
    slope: float = 0.0,
) -> np.ndarray:
    """Apply ALE over neighbouring bins while allowing a slope.

    Parameters
    ----------
    Sxx : np.ndarray
        Input magnitude spectrogram.
    track_width : int, optional
        Number of neighbouring bins to include when averaging. Default ``5``.
    delay : int, optional
        ALE delay parameter. Default ``3``.
    mu : float, optional
        Adaptation rate. Default ``0.01``.
    filter_order : int, optional
        Adaptive filter order. Default ``32``.
    slope : float, optional
        Linear slope used when calling :func:`apply_ale`. Default ``0.0``.
    """
    half = track_width // 2
    weights = np.exp(-0.5 * (np.arange(track_width) - half) ** 2)
    weights /= weights.sum()
    padded = np.pad(Sxx, ((half, half), (0, 0)), mode="reflect")
    out = np.zeros_like(Sxx)
    for i in range(Sxx.shape[0]):
        win = padded[i : i + track_width]
        avg = np.average(win, axis=0, weights=weights)
        filt = apply_ale(
            avg,
            delay=delay,
            mu=mu,
            filter_order=filter_order,
            slope=slope,
            freq_domain=False,
            return_error=False,
        )
        out[i] = filt
    return out


def apply_ale_2d_doppler_wave(
    x: np.ndarray,
    delay: int = 3,
    mu: float = 0.01,
    filter_order: int = 32,
    slope: float = 0.0,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    """Convenience wrapper applying ALE 2D Doppler in the STFT domain.

    The ``slope`` parameter is forwarded to :func:`apply_ale_2d_doppler` so
    drifting tones can be followed more closely.
    """
    f, t, Zxx = stft(x, nperseg=n_fft, noverlap=n_fft - hop_length)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    Sxx_f = apply_ale_2d_doppler(
        mag,
        delay=delay,
        mu=mu,
        filter_order=filter_order,
        slope=slope,
    )
    Zxx_f = Sxx_f * np.exp(1j * phase)
    _, out = istft(Zxx_f, nperseg=n_fft, noverlap=n_fft - hop_length)
    if len(out) > len(x):
        out = out[: len(x)]
    else:
        out = np.pad(out, (0, len(x) - len(out)))
    return out


def apply_wiener_adaptive_2d(Sxx: np.ndarray, size: tuple = (5, 5)) -> np.ndarray:
    """2-D Wiener filter with local noise estimation."""
    from skimage.restoration import wiener

    return wiener(Sxx, size)


def apply_track_following_filter(
    Sxx: np.ndarray,
    track_detector=None,
    enhancement_factor: float = 2.0,
) -> np.ndarray:
    """Strengthen spectrogram along detected tracks."""
    from scipy.ndimage import label

    if track_detector is None:
        thresh = np.percentile(Sxx, 90)
        binary = Sxx > thresh
        labeled, n = label(binary)
        tracks = []
        for i in range(1, n + 1):
            idx = np.where(labeled == i)
            tracks.append({"indices": idx})
    else:
        times = np.arange(Sxx.shape[1])
        freqs = np.arange(Sxx.shape[0])
        tracks = track_detector.auto_detect(Sxx, times, freqs)

    mask = np.zeros_like(Sxx)
    for tr in tracks:
        ti, fi = tr["indices"] if isinstance(tr, dict) else (tr[0], tr[1])
        for t_idx, f_idx in zip(ti, fi):
            for df in range(-3, 4):
                r = f_idx + df
                if 0 <= r < Sxx.shape[0]:
                    weight = np.exp(-0.5 * (df / 2) ** 2)
                    mask[r, t_idx] += weight
    mask = np.clip(mask, 0, 1)
    return Sxx * (1 + (enhancement_factor - 1) * mask)


def enhance_doppler_tracks(
    x: np.ndarray,
    fs: int,
    method: str = "combined",
    track_detection: bool = True,
    enhancement_factor: float = 2.0,
) -> np.ndarray:
    """Combined filtering optimized for Doppler tracks."""
    nperseg = min(1024, len(x) // 8)
    f, t, Zxx = stft(x, fs=fs, nperseg=nperseg)
    Sxx = np.abs(Zxx) ** 2
    if method == "track_only":
        if track_detection:
            Sxx = apply_track_following_filter(Sxx, enhancement_factor=enhancement_factor)
    else:
        if track_detection:
            Sxx = apply_track_following_filter(Sxx, enhancement_factor=enhancement_factor)
        Sxx = apply_wiener_adaptive_2d(Sxx)
        Sxx = apply_tv_denoising_doppler(Sxx)
    phase = np.angle(Zxx)
    Zxx_enh = np.sqrt(Sxx) * np.exp(1j * phase)
    _, out = istft(Zxx_enh, fs=fs, nperseg=nperseg)
    return out[: len(x)]
