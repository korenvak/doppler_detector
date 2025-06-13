from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QMessageBox
)
import numpy as np
from scipy.signal import butter, sosfilt
from spectrogram_gui.utils.spectrogram_utils import compute_spectrogram


class FilterDialog(QDialog):
    """
    Simple high-pass, low-pass or band-pass filter.
    """
    def __init__(self, main_window, mode="bandpass"):
        super().__init__(main_window)
        self.main   = main_window
        self.mode   = mode
        self.setWindowTitle(f"{mode.capitalize()} Filter")
        self.resize(360, 160)

        layout = QVBoxLayout(self)
        freq_layout = QHBoxLayout()

        if mode != "lowpass":
            freq_layout.addWidget(QLabel("Low cutoff (Hz):"))
            self.low_edit = QLineEdit("100")
            freq_layout.addWidget(self.low_edit)

        if mode != "highpass":
            freq_layout.addWidget(QLabel("High cutoff (Hz):"))
            self.high_edit = QLineEdit("3000")
            freq_layout.addWidget(self.high_edit)

        layout.addLayout(freq_layout)

        btns = QHBoxLayout()
        btns.addWidget(QPushButton("Cancel", clicked=self.reject))
        btns.addWidget(QPushButton("Apply", clicked=self.apply_filter))
        layout.addLayout(btns)

    def apply_filter(self):
        try:
            if self.mode != "lowpass":
                low = float(self.low_edit.text())
            if self.mode != "highpass":
                high = float(self.high_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid", "Enter valid cutoffs.")
            return

        sel = self.main.canvas.selected_range
        if sel is None:
            # apply to entire duration when no range selected
            t0, t1 = self.main.canvas.times[0], self.main.canvas.times[-1]
        else:
            t0, t1 = sel

        wave, sr = self.main.audio_player.get_waveform_copy(return_sr=True)
        total = len(wave)/sr
        i0, i1 = int((t0/total)*len(wave)), int((t1/total)*len(wave))
        if i1 <= i0:
            QMessageBox.warning(self, "Invalid Range", "Selected range is invalid.")
            return

        # design & apply
        if self.mode == "highpass":
            sos = butter(4, low, btype="highpass", fs=sr, output="sos")
        elif self.mode == "lowpass":
            sos = butter(4, high, btype="lowpass", fs=sr, output="sos")
        else:
            sos = butter(4, [low, high], btype="bandpass", fs=sr, output="sos")

        seg = wave[i0:i1].copy()
        filtered = sosfilt(sos, seg)
        new_wave = wave.copy(); new_wave[i0:i1] = filtered
        self.main.audio_player.replace_waveform(new_wave)

        # replot
        freqs, times, Sxx, _ = compute_spectrogram(
            new_wave, sr, "", params=self.main.spectrogram_params
        )
        self.main.canvas.plot_spectrogram(freqs, times, Sxx, self.main.canvas.start_time)
        self.accept()
