# File: personal/Koren/spectrogram_gui/gui/detector_params_dialog.py

from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QSpinBox, QDoubleSpinBox,
    QDialogButtonBox, QHBoxLayout, QComboBox
)
from PyQt5.QtCore import Qt

class DetectorParamsDialog(QDialog):
    def __init__(self, parent=None, detector=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Parameters")
        self.detector = detector

        layout = QFormLayout(self)

        self.method_box = QComboBox()
        self.method_box.addItems(["Peaks", "Threshold"])
        if getattr(detector, "detection_method", "peaks") == "threshold":
            self.method_box.setCurrentIndex(1)
        layout.addRow("Detection Method:", self.method_box)



        # 3) freq_min
        self.freq_min_spin = QSpinBox()
        self.freq_min_spin.setRange(0, 20000)
        self.freq_min_spin.setValue(detector.freq_min)
        layout.addRow("Min Frequency [Hz]:", self.freq_min_spin)

        # 4) freq_max
        self.freq_max_spin = QSpinBox()
        self.freq_max_spin.setRange(0, 20000)
        self.freq_max_spin.setValue(detector.freq_max)
        layout.addRow("Max Frequency [Hz]:", self.freq_max_spin)

        # 5) power_threshold
        self.power_thresh_spin = QDoubleSpinBox()
        self.power_thresh_spin.setRange(0.01, 1.00)
        self.power_thresh_spin.setSingleStep(0.01)
        self.power_thresh_spin.setDecimals(2)
        self.power_thresh_spin.setValue(detector.power_threshold)
        layout.addRow("Power Threshold:", self.power_thresh_spin)

        # 6) peak_prominence
        self.peak_prom_spin = QDoubleSpinBox()
        self.peak_prom_spin.setRange(0.001, 1.00)
        self.peak_prom_spin.setSingleStep(0.001)
        self.peak_prom_spin.setDecimals(3)
        self.peak_prom_spin.setValue(detector.peak_prominence)
        layout.addRow("Peak Prominence:", self.peak_prom_spin)

        # 7) max_gap_frames
        self.max_gap_spin = QSpinBox()
        self.max_gap_spin.setRange(0, 100)
        self.max_gap_spin.setValue(detector.max_gap_frames)
        layout.addRow("Max Gap Frames:", self.max_gap_spin)

        # 8) gap_power_factor
        self.gap_power_spin = QDoubleSpinBox()
        self.gap_power_spin.setRange(0.1, 1.0)
        self.gap_power_spin.setSingleStep(0.1)
        self.gap_power_spin.setValue(detector.gap_power_factor)
        layout.addRow("Gap Power Factor:", self.gap_power_spin)

        # 9) gap_prominence_factor
        self.gap_prom_spin2 = QDoubleSpinBox()
        self.gap_prom_spin2.setRange(0.1, 1.0)
        self.gap_prom_spin2.setSingleStep(0.1)
        self.gap_prom_spin2.setValue(detector.gap_prominence_factor)
        layout.addRow("Gap Prominence Factor:", self.gap_prom_spin2)

        # 10) max_freq_jump_hz
        self.max_jump_spin = QDoubleSpinBox()
        self.max_jump_spin.setRange(0.1, 1000.0)
        self.max_jump_spin.setSingleStep(1.0)
        self.max_jump_spin.setValue(detector.max_freq_jump_hz)
        layout.addRow("Max Freq Jump [Hz]:", self.max_jump_spin)

        # 11) gap_max_jump_hz
        self.gap_max_jump_spin = QDoubleSpinBox()
        self.gap_max_jump_spin.setRange(0.1, 1000.0)
        self.gap_max_jump_spin.setSingleStep(1.0)
        self.gap_max_jump_spin.setValue(detector.gap_max_jump_hz)
        layout.addRow("Gap Max Jump [Hz]:", self.gap_max_jump_spin)

        # 12) max_peaks_per_frame
        self.max_peaks_spin = QSpinBox()
        self.max_peaks_spin.setRange(1, 100)
        self.max_peaks_spin.setValue(detector.max_peaks_per_frame)
        layout.addRow("Max Peaks/Frame:", self.max_peaks_spin)

        # 13) min_track_length_frames
        self.min_length_spin = QSpinBox()
        self.min_length_spin.setRange(1, 1000)
        self.min_length_spin.setValue(detector.min_track_length_frames)
        layout.addRow("Min Track Length [frames]:", self.min_length_spin)

        # 14) min_track_avg_power
        self.min_avg_power_spin = QDoubleSpinBox()
        self.min_avg_power_spin.setRange(0.01, 1.00)
        self.min_avg_power_spin.setSingleStep(0.01)
        self.min_avg_power_spin.setDecimals(2)
        self.min_avg_power_spin.setValue(detector.min_track_avg_power)
        layout.addRow("Min Track Avg Power:", self.min_avg_power_spin)

        # 15) max_track_freq_std_hz
        self.max_std_spin = QDoubleSpinBox()
        self.max_std_spin.setRange(0.1, 1000.0)
        self.max_std_spin.setSingleStep(1.0)
        self.max_std_spin.setValue(detector.max_track_freq_std_hz)
        layout.addRow("Max Track Freq Std [Hz]:", self.max_std_spin)

        # 16) merge_gap_frames
        self.merge_gap_spin = QSpinBox()
        self.merge_gap_spin.setRange(0, 500)
        self.merge_gap_spin.setValue(detector.merge_gap_frames)
        layout.addRow("Merge Gap Frames:", self.merge_gap_spin)

        # 17) merge_max_freq_diff_hz
        self.merge_freq_diff_spin = QDoubleSpinBox()
        self.merge_freq_diff_spin.setRange(0.1, 1000.0)
        self.merge_freq_diff_spin.setSingleStep(1.0)
        self.merge_freq_diff_spin.setValue(detector.merge_max_freq_diff_hz)
        layout.addRow("Merge Max Freq Diff [Hz]:", self.merge_freq_diff_spin)

        # 18) smooth_sigma
        self.smooth_sigma_spin = QDoubleSpinBox()
        self.smooth_sigma_spin.setRange(0.1, 10.0)
        self.smooth_sigma_spin.setSingleStep(0.1)
        self.smooth_sigma_spin.setValue(detector.smooth_sigma)
        layout.addRow("Smooth Sigma:", self.smooth_sigma_spin)

        # 19) median_filter_size
        self.median_h_spin = QSpinBox()
        self.median_h_spin.setRange(1, 11)
        self.median_h_spin.setValue(detector.median_filter_size[0])
        self.median_w_spin = QSpinBox()
        self.median_w_spin.setRange(1, 11)
        self.median_w_spin.setValue(detector.median_filter_size[1])
        mf_layout = QHBoxLayout()
        mf_layout.addWidget(self.median_h_spin)
        mf_layout.addWidget(self.median_w_spin)
        layout.addRow("Median Filter Size (h × w):", mf_layout)

        # OK / Cancel
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def accept(self):
        d = self.detector
        d.freq_min = self.freq_min_spin.value()
        d.freq_max = self.freq_max_spin.value()
        d.power_threshold = self.power_thresh_spin.value()
        d.peak_prominence = self.peak_prom_spin.value()
        d.max_gap_frames = self.max_gap_spin.value()
        d.gap_power_factor = self.gap_power_spin.value()
        d.gap_prominence_factor = self.gap_prom_spin2.value()
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
        d.detection_method = "threshold" if self.method_box.currentIndex() == 1 else "peaks"
        super().accept()
