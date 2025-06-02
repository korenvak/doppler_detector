from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QSpinBox, QDoubleSpinBox, QDialogButtonBox, QHBoxLayout
)
from PyQt5.QtCore import Qt

class ParametersDialog(QDialog):
    def __init__(self, parent=None, detector=None):
        """
        Dialog for editing all detection parameters.
        The detector instance is passed to pre-fill defaults and to update directly.
        """
        super().__init__(parent)
        self.setWindowTitle("Detection Parameters")
        self.detector = detector

        layout = QFormLayout(self)

        # 1) nperseg
        self.nperseg_spin = QSpinBox()
        self.nperseg_spin.setRange(64, 16384)
        self.nperseg_spin.setSingleStep(64)
        self.nperseg_spin.setValue(detector.nperseg)
        self.nperseg_spin.setToolTip("Number of samples per FFT segment (nperseg)")
        layout.addRow("Nperseg:", self.nperseg_spin)

        # 2) noverlap
        self.noverlap_spin = QSpinBox()
        self.noverlap_spin.setRange(0, 16384)
        self.noverlap_spin.setSingleStep(64)
        self.noverlap_spin.setValue(detector.noverlap)
        self.noverlap_spin.setToolTip("Number of overlapping samples between segments (noverlap)")
        layout.addRow("Noverlap:", self.noverlap_spin)

        # 3) freq_min
        self.freq_min_spin = QSpinBox()
        self.freq_min_spin.setRange(0, 20000)
        self.freq_min_spin.setValue(detector.freq_min)
        self.freq_min_spin.setToolTip("Lower bound of frequency range (Hz)")
        layout.addRow("Min Frequency [Hz]:", self.freq_min_spin)

        # 4) freq_max
        self.freq_max_spin = QSpinBox()
        self.freq_max_spin.setRange(0, 20000)
        self.freq_max_spin.setValue(detector.freq_max)
        self.freq_max_spin.setToolTip("Upper bound of frequency range (Hz)")
        layout.addRow("Max Frequency [Hz]:", self.freq_max_spin)

        # 5) power_threshold
        self.power_thresh_spin = QDoubleSpinBox()
        self.power_thresh_spin.setRange(0.01, 1.00)
        self.power_thresh_spin.setSingleStep(0.01)
        self.power_thresh_spin.setDecimals(2)
        self.power_thresh_spin.setValue(detector.power_threshold)
        self.power_thresh_spin.setToolTip("Minimum power threshold (0.01–1.00)")
        layout.addRow("Power Threshold:", self.power_thresh_spin)

        # 6) peak_prominence (allow three decimals)
        self.peak_prom_spin = QDoubleSpinBox()
        self.peak_prom_spin.setRange(0.001, 1.00)
        self.peak_prom_spin.setSingleStep(0.001)
        self.peak_prom_spin.setDecimals(3)
        self.peak_prom_spin.setValue(detector.peak_prominence)
        self.peak_prom_spin.setToolTip("Minimum peak prominence (0.001–1.00)")
        layout.addRow("Peak Prominence:", self.peak_prom_spin)

        # 7) max_gap_frames
        self.max_gap_spin = QSpinBox()
        self.max_gap_spin.setRange(0, 100)
        self.max_gap_spin.setValue(detector.max_gap_frames)
        self.max_gap_spin.setToolTip("Maximum allowed gap (in frames) when linking peaks")
        layout.addRow("Max Gap Frames:", self.max_gap_spin)

        # 8) gap_power_factor
        self.gap_power_spin = QDoubleSpinBox()
        self.gap_power_spin.setRange(0.1, 1.0)
        self.gap_power_spin.setSingleStep(0.1)
        self.gap_power_spin.setValue(detector.gap_power_factor)
        self.gap_power_spin.setToolTip("Factor applied to power threshold when in a gap")
        layout.addRow("Gap Power Factor:", self.gap_power_spin)

        # 9) gap_prominence_factor
        self.gap_prom_spin = QDoubleSpinBox()
        self.gap_prom_spin.setRange(0.1, 1.0)
        self.gap_prom_spin.setSingleStep(0.1)
        self.gap_prom_spin.setValue(detector.gap_prominence_factor)
        self.gap_prom_spin.setToolTip("Factor applied to prominence when in a gap")
        layout.addRow("Gap Prominence Factor:", self.gap_prom_spin)

        # 10) max_freq_jump_hz
        self.max_jump_spin = QDoubleSpinBox()
        self.max_jump_spin.setRange(0.1, 1000.0)
        self.max_jump_spin.setSingleStep(1.0)
        self.max_jump_spin.setValue(detector.max_freq_jump_hz)
        self.max_jump_spin.setToolTip("Maximum allowed frequency jump (Hz) between frames")
        layout.addRow("Max Freq Jump [Hz]:", self.max_jump_spin)

        # 11) gap_max_jump_hz
        self.gap_max_jump_spin = QDoubleSpinBox()
        self.gap_max_jump_spin.setRange(0.1, 1000.0)
        self.gap_max_jump_spin.setSingleStep(1.0)
        self.gap_max_jump_spin.setValue(detector.gap_max_jump_hz)
        self.gap_max_jump_spin.setToolTip("Max freq jump (Hz) allowed when in a gap")
        layout.addRow("Gap Max Jump [Hz]:", self.gap_max_jump_spin)

        # 12) max_peaks_per_frame
        self.max_peaks_spin = QSpinBox()
        self.max_peaks_spin.setRange(1, 100)
        self.max_peaks_spin.setValue(detector.max_peaks_per_frame)
        self.max_peaks_spin.setToolTip("Maximum number of peaks to keep per frame")
        layout.addRow("Max Peaks/Frame:", self.max_peaks_spin)

        # 13) min_track_length_frames
        self.min_length_spin = QSpinBox()
        self.min_length_spin.setRange(1, 1000)
        self.min_length_spin.setValue(detector.min_track_length_frames)
        self.min_length_spin.setToolTip("Minimum number of consecutive frames for a valid track")
        layout.addRow("Min Track Length [frames]:", self.min_length_spin)

        # 14) min_track_avg_power
        self.min_avg_power_spin = QDoubleSpinBox()
        self.min_avg_power_spin.setRange(0.01, 1.00)
        self.min_avg_power_spin.setSingleStep(0.01)
        self.min_avg_power_spin.setDecimals(2)
        self.min_avg_power_spin.setValue(detector.min_track_avg_power)
        self.min_avg_power_spin.setToolTip("Minimum average power for a track")
        layout.addRow("Min Track Avg Power:", self.min_avg_power_spin)

        # 15) max_track_freq_std_hz
        self.max_std_spin = QDoubleSpinBox()
        self.max_std_spin.setRange(0.1, 1000.0)
        self.max_std_spin.setSingleStep(1.0)
        self.max_std_spin.setValue(detector.max_track_freq_std_hz)
        self.max_std_spin.setToolTip("Maximum standard deviation (Hz) of frequencies in a track")
        layout.addRow("Max Track Freq Std [Hz]:", self.max_std_spin)

        # 16) merge_gap_frames
        self.merge_gap_spin = QSpinBox()
        self.merge_gap_spin.setRange(0, 500)
        self.merge_gap_spin.setValue(detector.merge_gap_frames)
        self.merge_gap_spin.setToolTip("Max time gap (frames) to merge adjacent tracks")
        layout.addRow("Merge Gap Frames:", self.merge_gap_spin)

        # 17) merge_max_freq_diff_hz
        self.merge_freq_diff_spin = QDoubleSpinBox()
        self.merge_freq_diff_spin.setRange(0.1, 1000.0)
        self.merge_freq_diff_spin.setSingleStep(1.0)
        self.merge_freq_diff_spin.setValue(detector.merge_max_freq_diff_hz)
        self.merge_freq_diff_spin.setToolTip("Max frequency difference (Hz) to merge tracks")
        layout.addRow("Merge Max Freq Diff [Hz]:", self.merge_freq_diff_spin)

        # 18) smooth_sigma
        self.smooth_sigma_spin = QDoubleSpinBox()
        self.smooth_sigma_spin.setRange(0.1, 10.0)
        self.smooth_sigma_spin.setSingleStep(0.1)
        self.smooth_sigma_spin.setValue(detector.smooth_sigma)
        self.smooth_sigma_spin.setToolTip("Gaussian smoothing σ applied to spectrogram")
        layout.addRow("Smooth Sigma:", self.smooth_sigma_spin)

        # 19) median_filter_size (height × width)
        self.median_h_spin = QSpinBox()
        self.median_h_spin.setRange(1, 11)
        self.median_h_spin.setValue(detector.median_filter_size[0])
        self.median_h_spin.setToolTip("Median filter kernel height (odd integer)")
        self.median_w_spin = QSpinBox()
        self.median_w_spin.setRange(1, 11)
        self.median_w_spin.setValue(detector.median_filter_size[1])
        self.median_w_spin.setToolTip("Median filter kernel width (odd integer)")
        mf_layout = QHBoxLayout()
        mf_layout.addWidget(self.median_h_spin)
        mf_layout.addWidget(self.median_w_spin)
        layout.addRow("Median Filter Size (h × w):", mf_layout)

        # OK / Cancel buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addRow(self.button_box)

    def accept(self):
        """
        Update detector parameters from the dialog’s inputs.
        """
        d = self.detector
        d.nperseg = self.nperseg_spin.value()
        d.noverlap = self.noverlap_spin.value()
        d.freq_min = self.freq_min_spin.value()
        d.freq_max = self.freq_max_spin.value()
        d.power_threshold = self.power_thresh_spin.value()
        d.peak_prominence = self.peak_prom_spin.value()
        d.max_gap_frames = self.max_gap_spin.value()
        d.gap_power_factor = self.gap_power_spin.value()
        d.gap_prominence_factor = self.gap_prom_spin.value()
        d.max_freq_jump_hz = self.max_jump_spin.value()
        d.gap_max_jump_hz = self.gap_max_jump_spin.value()
        d.max_peaks_per_frame = self.max_peaks_spin.value()
        d.min_track_length_frames = self.min_length_spin.value()
        d.min_track_avg_power = self.min_avg_power_spin.value()
        d.max_track_freq_std_hz = self.max_std_spin.value()
        d.merge_gap_frames = self.merge_gap_spin.value()
        d.merge_max_freq_diff_hz = self.merge_freq_diff_spin.value()
        d.smooth_sigma = self.smooth_sigma_spin.value()
        d.median_filter_size = (self.median_h_spin.value(), self.median_w_spin.value())

        super().accept()
