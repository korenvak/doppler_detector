from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox
import numpy as np
from spectrogram_gui.utils.spectrogram_utils import compute_spectrogram


class NormalizeDialog(QDialog):
    """
    Dialog to normalize audio segment to maximum absolute amplitude of 1.0.
    """
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("Normalize Selection")

        layout = QVBoxLayout()
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Normalize selection to full scale?"))
        layout.addLayout(h_layout)

        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Normalize")
        self.apply_btn.clicked.connect(self.apply_normalize)
        btn_layout.addWidget(self.apply_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def apply_normalize(self):
        selected = self.main_window.canvas.selected_range
        if not selected:
            t0, t1 = self.main_window.canvas.times[0], self.main_window.canvas.times[-1]
        else:
            t0, t1 = selected

        # Backup for undo
        wave = self.main_window.audio_player.get_waveform_copy()
        prev_sxx = self.main_window.canvas.Sxx_raw.copy()
        prev_times = self.main_window.canvas.times.copy()
        prev_freqs = self.main_window.canvas.freqs.copy()
        self.main_window.undo_stack.append((wave, prev_sxx, prev_times, prev_freqs))
        self.main_window.undo_btn.setEnabled(True)

        # Normalize waveform segment
        wave_data, sr = self.main_window.audio_player.get_waveform(), self.main_window.audio_player.sr
        n_total = len(wave_data)
        duration = n_total / sr
        idx0 = int((t0 / duration) * n_total)
        idx1 = int((t1 / duration) * n_total)
        idx0, idx1 = max(0, min(idx0, n_total)), max(0, min(idx1, n_total))
        if idx1 <= idx0:
            QMessageBox.warning(self, "Range Error", "Selected range is invalid.")
            return

        segment = wave_data[idx0:idx1]
        max_val = np.max(np.abs(segment))
        if max_val == 0:
            QMessageBox.warning(self, "Normalize Error", "Segment is silent.")
            return

        new_wave = wave_data.copy()
        new_wave[idx0:idx1] = segment / max_val
        self.main_window.audio_player.replace_waveform(new_wave)

        # Recompute spectrogram
        freqs, times, Sxx, _ = compute_spectrogram(
            new_wave, sr, "", params=self.main_window.spectrogram_params
        )
        self.main_window.canvas.plot_spectrogram(
            freqs, times, Sxx, self.main_window.canvas.start_time
        )

        self.accept()