import numpy as np
from scipy.signal import stft, istft
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QPushButton,
    QMessageBox,
)
from spectrogram_gui.utils.spectrogram_utils import compute_spectrogram
from spectrogram_gui.utils.filter_utils import (
    apply_nlms,
    apply_ale,
    apply_wiener,
    apply_gaussian,
    apply_median,
    apply_gabor,
)





class CombinedFilterDialog(QDialog):
    """
    Dialog to apply NLMS, ALE, Wiener and other smoothing filters
    to the selected time-range in the spectrogram GUI.
    """
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main = main_window
        self.setWindowTitle("Apply Filters")
        self.resize(460, 220)

        layout = QVBoxLayout(self)

        # --- checkboxes ---
        self.nlms_chk = QCheckBox("Enable NLMS")
        self.ale_chk = QCheckBox("Enable ALE")
        self.wiener_chk = QCheckBox("Enable Wiener Adaptive")
        self.gauss_chk = QCheckBox("Gaussian Smooth")
        self.median_chk = QCheckBox("Median Filter")
        self.gabor_chk = QCheckBox("Gabor Filter")
        layout.addWidget(self.nlms_chk)
        layout.addWidget(self.ale_chk)
        layout.addWidget(self.wiener_chk)
        layout.addWidget(self.gauss_chk)
        layout.addWidget(self.median_chk)
        layout.addWidget(self.gabor_chk)

        # --- parameter controls ---
        p_layout = QHBoxLayout()
        p_layout.addWidget(QLabel("NLMS μ (0–1):"))
        self.nlms_spin = QDoubleSpinBox()
        self.nlms_spin.setRange(0.0001, 1.0)
        self.nlms_spin.setSingleStep(0.001)
        self.nlms_spin.setValue(0.01)
        p_layout.addWidget(self.nlms_spin)

        p_layout.addWidget(QLabel("ALE μ (0–1):"))
        self.ale_spin = QDoubleSpinBox()
        self.ale_spin.setRange(0.0001, 1.0)
        self.ale_spin.setSingleStep(0.001)
        self.ale_spin.setValue(0.01)
        p_layout.addWidget(self.ale_spin)

        p_layout.addWidget(QLabel("ALE Delay:"))
        self.ale_delay_spin = QSpinBox()
        self.ale_delay_spin.setRange(0, 100)
        self.ale_delay_spin.setValue(0)
        self.ale_delay_spin.setSpecialValueText("Auto")
        self.ale_delay_spin.setToolTip("0 = automatic delay search")
        p_layout.addWidget(self.ale_delay_spin)

        p_layout.addWidget(QLabel("ALE Slope:"))
        self.ale_slope_spin = QDoubleSpinBox()
        self.ale_slope_spin.setRange(-1.0, 1.0)
        self.ale_slope_spin.setSingleStep(0.01)
        self.ale_slope_spin.setValue(0.0)
        p_layout.addWidget(self.ale_slope_spin)


        p_layout.addWidget(QLabel("Wiener Noise (dB):"))
        self.wiener_spin = QDoubleSpinBox()
        self.wiener_spin.setRange(-60, 0)
        self.wiener_spin.setSingleStep(1)
        self.wiener_spin.setValue(-20)
        p_layout.addWidget(self.wiener_spin)

        p_layout.addWidget(QLabel("Gauss σ:"))
        self.gauss_spin = QDoubleSpinBox()
        self.gauss_spin.setRange(0.1, 10.0)
        self.gauss_spin.setSingleStep(0.1)
        self.gauss_spin.setValue(1.0)
        p_layout.addWidget(self.gauss_spin)

        p_layout.addWidget(QLabel("Median k:"))
        self.median_spin = QSpinBox()
        self.median_spin.setRange(1, 99)
        self.median_spin.setSingleStep(2)
        self.median_spin.setValue(3)
        p_layout.addWidget(self.median_spin)

        p_layout.addWidget(QLabel("Gabor f:"))
        self.gabor_freq_spin = QDoubleSpinBox()
        self.gabor_freq_spin.setRange(0.01, 0.5)
        self.gabor_freq_spin.setSingleStep(0.01)
        self.gabor_freq_spin.setValue(0.1)
        p_layout.addWidget(self.gabor_freq_spin)

        p_layout.addWidget(QLabel("Gabor σ:"))
        self.gabor_sigma_spin = QDoubleSpinBox()
        self.gabor_sigma_spin.setRange(0.5, 10.0)
        self.gabor_sigma_spin.setSingleStep(0.1)
        self.gabor_sigma_spin.setValue(2.0)
        p_layout.addWidget(self.gabor_sigma_spin)

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
        self.main.add_undo_action(("waveform", (
            prev_wave.copy(), prev_sxx, prev_times, prev_freqs, prev_start
        )))

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
            delay_val = self.ale_delay_spin.value()
            delay = None if delay_val == 0 else delay_val
            out = apply_ale(
                out,
                delay=delay,
                mu=self.ale_spin.value(),
                filter_order=order,
                slope=self.ale_slope_spin.value(),
                freq_domain=True,
            )
        if self.wiener_chk.isChecked():
            out = apply_wiener(out, noise_db=self.wiener_spin.value())

        if self.gauss_chk.isChecked():
            out = apply_gaussian(out, sigma=self.gauss_spin.value())

        if self.median_chk.isChecked():
            out = apply_median(out, size=self.median_spin.value())

        if self.gabor_chk.isChecked():
            out = apply_gabor(
                out,
                freq=self.gabor_freq_spin.value(),
                sigma=self.gabor_sigma_spin.value(),
            )

        # 5) write back & replot
        new_wave = wave.copy()
        new_wave[i0:i1] = out
        self.main.audio_player.replace_waveform(new_wave)

        freqs, times, Sxx, _ = compute_spectrogram(
            new_wave, sr, "", params=self.main.spectrogram_params
        )
        self.main.canvas.plot_spectrogram(freqs, times, Sxx, self.main.canvas.start_time, maintain_view=True)

        self.accept()
