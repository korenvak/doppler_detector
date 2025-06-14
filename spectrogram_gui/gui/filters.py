from __future__ import annotations

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QMessageBox,
    QStackedWidget,
    QWidget,
)
from spectrogram_gui.utils.spectrogram_utils import compute_spectrogram
from spectrogram_gui.utils.filter_utils import (
    apply_nlms,
    apply_ale_2d_doppler_wave,
    apply_gaussian,
    apply_median,
    apply_gabor,
)





class CombinedFilterDialog(QDialog):
    """Dialog to apply a single adaptive or smoothing filter."""

    def __init__(self, main_window, initial_filter: str | None = None):
        super().__init__(main_window)
        self.main = main_window
        self.setWindowTitle("Apply Filter")
        self.resize(360, 200)

        layout = QVBoxLayout(self)

        self.filter_box = QComboBox()
        self.filter_box.addItems([
            "NLMS",
            "ALE 2D Doppler",
            "Wiener Adaptive",
            "Gaussian",
            "Median",
            "Gabor",
            "TV Denoise Doppler",
            "Track Follow",
            "Enhance Doppler",
        ])
        layout.addWidget(self.filter_box)

        self.stack = QStackedWidget()

        # --- NLMS params ---
        w = QHBoxLayout()
        nlms_widget = QWidget()
        nlms_widget.setLayout(w)
        w.addWidget(QLabel("μ:"))
        self.nlms_spin = QDoubleSpinBox()
        self.nlms_spin.setRange(0.0001, 1.0)
        self.nlms_spin.setSingleStep(0.001)
        self.nlms_spin.setValue(0.01)
        w.addWidget(self.nlms_spin)
        self.stack.addWidget(nlms_widget)

        # --- ALE params ---
        w = QHBoxLayout()
        ale_widget = QWidget()
        ale_widget.setLayout(w)
        w.addWidget(QLabel("μ:"))
        self.ale_spin = QDoubleSpinBox()
        self.ale_spin.setRange(0.0001, 1.0)
        self.ale_spin.setSingleStep(0.001)
        self.ale_spin.setValue(0.01)
        w.addWidget(self.ale_spin)
        w.addWidget(QLabel("Delay:"))
        self.ale_delay_spin = QSpinBox()
        self.ale_delay_spin.setRange(0, 100)
        self.ale_delay_spin.setValue(0)
        self.ale_delay_spin.setSpecialValueText("Auto")
        w.addWidget(self.ale_delay_spin)
        w.addWidget(QLabel("Slope:"))
        self.ale_slope_spin = QDoubleSpinBox()
        self.ale_slope_spin.setRange(-1.0, 1.0)
        self.ale_slope_spin.setSingleStep(0.01)
        self.ale_slope_spin.setValue(0.0)
        w.addWidget(self.ale_slope_spin)
        self.stack.addWidget(ale_widget)

        # --- Wiener params ---
        w = QHBoxLayout()
        wien_widget = QWidget()
        wien_widget.setLayout(w)
        w.addWidget(QLabel("Noise dB:"))
        self.wiener_spin = QDoubleSpinBox()
        self.wiener_spin.setRange(-60, 0)
        self.wiener_spin.setSingleStep(1)
        self.wiener_spin.setValue(-20)
        w.addWidget(self.wiener_spin)
        self.stack.addWidget(wien_widget)

        # --- Gaussian params ---
        w = QHBoxLayout()
        gauss_widget = QWidget()
        gauss_widget.setLayout(w)
        w.addWidget(QLabel("σ:"))
        self.gauss_spin = QDoubleSpinBox()
        self.gauss_spin.setRange(0.1, 10.0)
        self.gauss_spin.setSingleStep(0.1)
        self.gauss_spin.setValue(1.0)
        w.addWidget(self.gauss_spin)
        self.stack.addWidget(gauss_widget)

        # --- Median params ---
        w = QHBoxLayout()
        med_widget = QWidget()
        med_widget.setLayout(w)
        w.addWidget(QLabel("k:"))
        self.median_spin = QSpinBox()
        self.median_spin.setRange(1, 99)
        self.median_spin.setSingleStep(2)
        self.median_spin.setValue(3)
        w.addWidget(self.median_spin)
        self.stack.addWidget(med_widget)

        # --- Gabor params ---
        w = QHBoxLayout()
        gabor_widget = QWidget()
        gabor_widget.setLayout(w)
        w.addWidget(QLabel("f:"))
        self.gabor_freq_spin = QDoubleSpinBox()
        self.gabor_freq_spin.setRange(0.01, 0.5)
        self.gabor_freq_spin.setSingleStep(0.01)
        self.gabor_freq_spin.setValue(0.1)
        w.addWidget(self.gabor_freq_spin)
        w.addWidget(QLabel("σ:"))
        self.gabor_sigma_spin = QDoubleSpinBox()
        self.gabor_sigma_spin.setRange(0.5, 10.0)
        self.gabor_sigma_spin.setSingleStep(0.1)
        self.gabor_sigma_spin.setValue(2.0)
        w.addWidget(self.gabor_sigma_spin)
        self.stack.addWidget(gabor_widget)

        # --- TV Denoise params ---
        w = QHBoxLayout()
        tv_widget = QWidget()
        tv_widget.setLayout(w)
        w.addWidget(QLabel("Weight:"))
        self.tv_weight_spin = QDoubleSpinBox()
        self.tv_weight_spin.setRange(0.01, 1.0)
        self.tv_weight_spin.setSingleStep(0.01)
        self.tv_weight_spin.setValue(0.1)
        w.addWidget(self.tv_weight_spin)
        self.stack.addWidget(tv_widget)

        # --- Track Follow params ---
        w = QHBoxLayout()
        tf_widget = QWidget()
        tf_widget.setLayout(w)
        w.addWidget(QLabel("Factor:"))
        self.track_factor_spin = QDoubleSpinBox()
        self.track_factor_spin.setRange(1.0, 5.0)
        self.track_factor_spin.setSingleStep(0.1)
        self.track_factor_spin.setValue(2.0)
        w.addWidget(self.track_factor_spin)
        self.stack.addWidget(tf_widget)

        # --- Enhance Doppler params ---
        w = QHBoxLayout()
        ed_widget = QWidget()
        ed_widget.setLayout(w)
        self.enhance_track_chk = QCheckBox("Detect Tracks")
        self.enhance_track_chk.setChecked(True)
        w.addWidget(self.enhance_track_chk)
        self.stack.addWidget(ed_widget)

        layout.addWidget(self.stack)

        self.filter_box.currentIndexChanged.connect(self.stack.setCurrentIndex)

        if initial_filter:
            idx = self.filter_box.findText(initial_filter)
            if idx >= 0:
                self.filter_box.setCurrentIndex(idx)
            self.filter_box.setVisible(False)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.apply_filter)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def apply_filter(self):
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

        # 4) apply selected filter
        order = min(32, len(seg))
        out = seg.copy()
        filt = self.filter_box.currentText()

        if filt == "NLMS":
            if len(out) < order:
                QMessageBox.warning(self, "Too Short", "Segment shorter than NLMS order.")
                return
            out = apply_nlms(out, mu=self.nlms_spin.value(), filter_order=order)
        elif filt == "ALE 2D Doppler":
            if len(out) < order:
                QMessageBox.warning(self, "Too Short", "Segment shorter than ALE order.")
                return
            delay_val = self.ale_delay_spin.value()
            delay = None if delay_val == 0 else delay_val
            from spectrogram_gui.utils.filter_utils import apply_ale_2d_doppler_wave
            out = apply_ale_2d_doppler_wave(
                out,
                delay=delay if delay is not None else 3,
                mu=self.ale_spin.value(),
                filter_order=order,
                slope=self.ale_slope_spin.value(),
            )
        elif filt == "Wiener Adaptive":
            from spectrogram_gui.utils.filter_utils import apply_wiener_adaptive
            out = apply_wiener_adaptive(out, window_size=1024)
        elif filt == "Gaussian":
            out = apply_gaussian(out, sigma=self.gauss_spin.value())
        elif filt == "Median":
            out = apply_median(out, size=self.median_spin.value())
        elif filt == "Gabor":
            out = apply_gabor(out, freq=self.gabor_freq_spin.value(), sigma=self.gabor_sigma_spin.value())
        elif filt == "TV Denoise Doppler":
            from spectrogram_gui.utils.filter_utils import apply_tv_denoising_doppler_wave
            out = apply_tv_denoising_doppler_wave(out, weight=self.tv_weight_spin.value())
        elif filt == "Track Follow":
            from spectrogram_gui.utils.filter_utils import enhance_doppler_tracks
            out = enhance_doppler_tracks(
                out,
                fs=sr,
                method="track_only",
                track_detection=True,
                enhancement_factor=self.track_factor_spin.value(),
            )
        elif filt == "Enhance Doppler":
            from spectrogram_gui.utils.filter_utils import enhance_doppler_tracks
            out = enhance_doppler_tracks(
                out,
                fs=sr,
                method="combined",
                track_detection=self.enhance_track_chk.isChecked(),
                enhancement_factor=self.track_factor_spin.value(),
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
