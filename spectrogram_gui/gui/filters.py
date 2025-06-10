import numpy as np
from scipy.signal import stft, istft
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QDoubleSpinBox, QPushButton, QMessageBox
)
from personal.Koren.spectrogram_gui.utils.spectrogram_utils import compute_spectrogram


import numpy as np

def apply_nlms(y, mu=0.1, filter_order=32, eps=1e-6):
    """
    Normalized LMS noise‐canceller along time axis.
    y: 1D audio waveform
    """
    N = len(y)
    # initialize
    w = np.zeros(filter_order)
    out = np.zeros_like(y)

    # run frame by frame
    for n in range(N):
        # only start once we have enough history
        if n >= filter_order:
            # get the last 'filter_order' samples
            u = y[n-filter_order:n]           # shape = (filter_order,)
            # prediction
            y_pred = np.dot(w, u)
            # error
            e = y[n] - y_pred
            # normalized step
            norm = np.dot(u, u) + eps
            w += (mu / norm) * e * u
            out[n] = e
        else:
            # before buffer fills, just passthrough
            out[n] = y[n]

    return out


import numpy as np

def apply_ale(y,
              order: int = 32,
              delay: int = 1,
              mu: float = 0.1,
              forgetting_factor: float = 1.0,
              eps: float = 1e-6):
    """
    Feedback-delay adaptive line enhancer (ALE) via leaky-LMS.
    - order: number of filter taps
    - delay: prediction horizon (in samples)
    - mu: adaptation rate
    - forgetting_factor: leaky-LMS leakage (1.0 = no leakage)
    - eps: regularization to avoid divide-by-zero
    """
    N = len(y)
    w = np.zeros(order, dtype=float)
    out = np.zeros_like(y, dtype=float)

    for n in range(N):
        if n >= order + delay:
            # always get exactly 'order' samples behind (n-delay-order .. n-delay-1)
            u = y[n - delay - order : n - delay]
            # if slicing gave fewer than order (shouldn't happen if n >= order+delay), just passthrough
            if u.shape[0] == order:
                # predicted value
                y_pred = np.dot(w, u)
                e = y[n] - y_pred
                norm_u = np.dot(u, u) + eps
                # leaky-LMS weight update
                w = forgetting_factor * w + (mu / norm_u) * e * u
                out[n] = e
            else:
                out[n] = y[n]
        else:
            out[n] = y[n]

    return out


def apply_wiener(
    x: np.ndarray,
    noise_db: float = -20,
    window_size: int = 1024,
    overlap: int = 512
) -> np.ndarray:
    """Wiener filter via spectral subtraction (STFT/ISTFT)."""
    noise_pow = 10 ** (noise_db / 10)
    f, t, Zxx = stft(x, nperseg=window_size, noverlap=overlap)
    Sxx = np.abs(Zxx) ** 2
    G = Sxx / (Sxx + noise_pow)
    Zxx_w = Zxx * G
    _, x_w = istft(Zxx_w, nperseg=window_size, noverlap=overlap)

    if len(x_w) > len(x):
        return x_w[:len(x)]
    return np.pad(x_w, (0, len(x) - len(x_w)))


class CombinedFilterDialog(QDialog):
    """
    Dialog to apply NLMS, ALE and/or Wiener adaptive filters
    to the selected time-range in the spectrogram GUI.
    """
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main = main_window
        self.setWindowTitle("Apply Filters")
        self.resize(400, 200)

        layout = QVBoxLayout(self)

        # --- checkboxes ---
        self.nlms_chk = QCheckBox("Enable NLMS")
        self.ale_chk = QCheckBox("Enable ALE")
        self.wiener_chk = QCheckBox("Enable Wiener Adaptive")
        layout.addWidget(self.nlms_chk)
        layout.addWidget(self.ale_chk)
        layout.addWidget(self.wiener_chk)

        # --- parameter controls ---
        p_layout = QHBoxLayout()
        p_layout.addWidget(QLabel("NLMS μ (0–1):"))
        self.nlms_spin = QDoubleSpinBox()
        self.nlms_spin.setRange(0.0001, 1.0)
        self.nlms_spin.setSingleStep(0.001)
        self.nlms_spin.setValue(0.01)
        p_layout.addWidget(self.nlms_spin)

        p_layout.addWidget(QLabel("ALE λ (0–1):"))
        self.ale_spin = QDoubleSpinBox()
        self.ale_spin.setRange(0.9, 1.0)
        self.ale_spin.setSingleStep(0.001)
        self.ale_spin.setValue(0.99)
        p_layout.addWidget(self.ale_spin)

        p_layout.addWidget(QLabel("Wiener Noise (dB):"))
        self.wiener_spin = QDoubleSpinBox()
        self.wiener_spin.setRange(-60, 0)
        self.wiener_spin.setSingleStep(1)
        self.wiener_spin.setValue(-20)
        p_layout.addWidget(self.wiener_spin)

        layout.addLayout(p_layout)

        # --- buttons ---
        btns = QHBoxLayout()
        btns.addStretch()
        btns.addWidget(QPushButton("Apply Filters", clicked=self.apply_filters))
        btns.addWidget(QPushButton("Cancel", clicked=self.reject))
        layout.addLayout(btns)

    def apply_filters(self):
        # 1) selection
        sel = self.main.canvas.selected_range
        if sel is None:
            # whole spectrogram if nothing selected
            t0, t1 = self.main.canvas.times[0], self.main.canvas.times[-1]
        else:
            t0, t1 = sel

        # 2) extract segment
        wave, sr = self.main.audio_player.get_waveform_copy(return_sr=True)
        total = len(wave) / sr
        i0 = int((t0 / total) * len(wave))
        i1 = int((t1 / total) * len(wave))
        if i1 <= i0:
            QMessageBox.warning(self, "Invalid Range", "Selected range is invalid.")
            return
        seg = wave[i0:i1].copy()

        # 3) push _full_ undo state (wave + spectrogram)
        prev_wave, _ = self.main.audio_player.get_waveform_copy(return_sr=True)
        prev_sxx   = self.main.canvas.current_sxx.copy()
        prev_times = self.main.canvas.current_times.copy()
        prev_freqs = self.main.canvas.current_freqs.copy()
        prev_start = self.main.canvas.start_time
        self.main.undo_stack.append((
            prev_wave.copy(), prev_sxx, prev_times, prev_freqs, prev_start
        ))
        self.main.undo_btn.setEnabled(True)

        # 4) apply each filter
        order = min(32, len(seg))
        out = seg.copy()

        if self.nlms_chk.isChecked():
            if len(out) < order:
                QMessageBox.warning(self, "Too Short", "Segment shorter than NLMS order.")
                return
            out = apply_nlms(out, mu=self.nlms_spin.value(), filter_order=order)

        if self.ale_chk.isChecked():
            if len(out) < order:
                QMessageBox.warning(self, "Too Short", "Segment shorter than ALE order.")
                return
            # basic ALE with fixed delay of 1 sample
            out = apply_ale(
                out,
                order=order,
                delay=1,
                mu=0.1,
                forgetting_factor=self.ale_spin.value()
            )
        if self.wiener_chk.isChecked():
            out = apply_wiener(out, noise_db=self.wiener_spin.value())

        # 5) write back & replot
        new_wave = wave.copy()
        new_wave[i0:i1] = out
        self.main.audio_player.replace_waveform(new_wave)

        freqs, times, Sxx, _ = compute_spectrogram(
            new_wave, sr, "", params=self.main.spectrogram_params
        )
        self.main.canvas.plot_spectrogram(freqs, times, Sxx, self.main.canvas.start_time)

        self.accept()
