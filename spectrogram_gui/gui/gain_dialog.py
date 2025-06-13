# File: gain_dialog.py

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
import numpy as np
from spectrogram_gui.utils.spectrogram_utils import compute_spectrogram


class GainDialog(QDialog):
    """
    Dialog to apply a gain factor to the selected segment.
    """

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Apply Gain to Selected Segment")
        self.resize(320, 120)

        layout = QVBoxLayout(self)

        # Gain factor input
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Gain factor:"))
        self.gain_edit = QLineEdit("2.0")
        gain_layout.addWidget(self.gain_edit)
        layout.addLayout(gain_layout)

        # Buttons: Apply + Cancel
        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.apply_btn.clicked.connect(self.apply_gain)
        self.cancel_btn.clicked.connect(self.reject)

    def apply_gain(self):
        try:
            gain_factor = float(self.gain_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid numerical gain.")
            return

        # Selected time range (use entire duration if none)
        sel = self.main_window.canvas.selected_range
        if sel is None:
            t0, t1 = self.main_window.canvas.times[0], self.main_window.canvas.times[-1]
        else:
            t0, t1 = sel

        # Current waveform + sr
        wave = self.main_window.audio_player.data
        sr = self.main_window.audio_player.sample_rate

        total_duration = len(wave) / sr
        idx0 = int(np.floor((t0 / total_duration) * len(wave)))
        idx1 = int(np.ceil((t1 / total_duration) * len(wave)))
        idx0 = max(0, min(idx0, len(wave) - 1))
        idx1 = max(0, min(idx1, len(wave)))

        if idx1 <= idx0:
            QMessageBox.warning(self, "Selection Error", "Selected time range is invalid.")
            return

        # backup for undo
        prev_sxx = self.main_window.canvas.Sxx_raw.copy()
        prev_times = self.main_window.canvas.times.copy()
        prev_freqs = self.main_window.canvas.freqs.copy()
        prev_start = self.main_window.canvas.start_time
        self.main_window.add_undo_action(("waveform", (wave.copy(), prev_sxx, prev_times, prev_freqs, prev_start)))

        # Apply gain
        new_wave = wave.copy()
        new_wave[idx0:idx1] *= gain_factor

        # Replace waveform in player
        self.main_window.audio_player.replace_waveform(new_wave)

        # Recompute spectrogram
        freqs, times, Sxx, _ = compute_spectrogram(
            new_wave, sr, "", params=self.main_window.spectrogram_params
        )
        self.main_window.canvas.plot_spectrogram(
            freqs, times, Sxx, self.main_window.canvas.start_time, maintain_view=True
        )

        self.accept()
