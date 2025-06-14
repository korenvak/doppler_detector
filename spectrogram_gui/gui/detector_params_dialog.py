# File: personal/Koren/spectrogram_gui/gui/detector_params_dialog.py

from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QSpinBox, QDoubleSpinBox,
    QDialogButtonBox, QComboBox, QLabel
)
from PyQt5.QtCore import Qt

class DetectorParamsDialog(QDialog):
    def __init__(self, parent=None, detector=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Parameters")
        self.detector = detector

        layout = QFormLayout(self)

        self.method_box = QComboBox()
        self.method_box.addItems(["Peaks", "Advanced"])
        method = getattr(detector, "detection_method", "peaks")
        if method == "advanced":
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
        self.peak_prom_label = QLabel("Peak Prominence:")
        self.peak_prom_spin = QDoubleSpinBox()
        self.peak_prom_spin.setRange(0.001, 1.00)
        self.peak_prom_spin.setSingleStep(0.001)
        self.peak_prom_spin.setDecimals(3)
        self.peak_prom_spin.setValue(detector.peak_prominence)
        layout.addRow(self.peak_prom_label, self.peak_prom_spin)

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
        self.max_peaks_label = QLabel("Max Peaks/Frame:")
        self.max_peaks_spin = QSpinBox()
        self.max_peaks_spin.setRange(1, 100)
        self.max_peaks_spin.setValue(detector.max_peaks_per_frame)
        layout.addRow(self.max_peaks_label, self.max_peaks_spin)

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

        # 18) Advanced mask percentile
        self.adv_thresh_label = QLabel("Mask Percentile:")
        self.adv_thresh_spin = QSpinBox()
        self.adv_thresh_spin.setRange(50, 100)
        self.adv_thresh_spin.setValue(detector.adv_threshold_percentile)
        layout.addRow(self.adv_thresh_label, self.adv_thresh_spin)

        # 19) Advanced line length
        self.adv_len_label = QLabel("Min Line Length:")
        self.adv_len_spin = QSpinBox()
        self.adv_len_spin.setRange(1, 1000)
        self.adv_len_spin.setValue(detector.adv_min_line_length)
        layout.addRow(self.adv_len_label, self.adv_len_spin)

        # 20) Advanced line gap
        self.adv_gap_label = QLabel("Line Gap:")
        self.adv_gap_spin = QSpinBox()
        self.adv_gap_spin.setRange(1, 100)
        self.adv_gap_spin.setValue(detector.adv_line_gap)
        layout.addRow(self.adv_gap_label, self.adv_gap_spin)


        # OK / Cancel
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self.method_box.currentIndexChanged.connect(self.update_visibility)
        self.update_visibility()


    def update_visibility(self):
        peaks = self.method_box.currentIndex() == 0
        advanced = self.method_box.currentIndex() == 1

        peak_widgets = [
            self.power_thresh_spin,
            self.peak_prom_label,
            self.peak_prom_spin,
            self.max_gap_spin,
            self.gap_power_spin,
            self.gap_prom_spin2,
            self.max_jump_spin,
            self.gap_max_jump_spin,
            self.max_peaks_label,
            self.max_peaks_spin,
        ]

        adv_widgets = [
            self.adv_thresh_label,
            self.adv_thresh_spin,
            self.adv_len_label,
            self.adv_len_spin,
            self.adv_gap_label,
            self.adv_gap_spin,
        ]

        for w in peak_widgets:
            w.setVisible(peaks)

        for w in adv_widgets:
            w.setVisible(advanced)

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
        d.adv_threshold_percentile = self.adv_thresh_spin.value()
        d.adv_min_line_length = self.adv_len_spin.value()
        d.adv_line_gap = self.adv_gap_spin.value()
        if self.method_box.currentIndex() == 1:
            d.detection_method = "advanced"
        else:
            d.detection_method = "peaks"
        super().accept()
