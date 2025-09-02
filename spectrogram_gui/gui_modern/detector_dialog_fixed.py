"""
Fixed Modern Detector Parameters Dialog with proper detector integration
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QDoubleSpinBox, QSpinBox, QPushButton,
    QGroupBox, QFormLayout, QDialogButtonBox
)
from PySide6.QtCore import Qt, Signal
import numpy as np


class ModernDetectorDialog(QDialog):
    """
    Modern detector configuration dialog that properly interfaces with DopplerDetector
    """
    detection_started = Signal(dict)
    detection_finished = Signal(object)
    
    def __init__(self, parent=None, detector=None, mode="peaks"):
        super().__init__(parent)
        self.detector = detector
        self.mode = mode
        self.setWindowTitle("Detection Parameters")
        self.setModal(True)
        self.setMinimumSize(500, 600)
        
        # Apply modern styling
        self.setStyleSheet("""
            QDialog {
                background: rgba(20, 20, 30, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
            }
            QGroupBox {
                color: rgba(255, 255, 255, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 16px;
                font-weight: 500;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
            QLabel {
                color: rgba(255, 255, 255, 0.8);
            }
            QSpinBox, QDoubleSpinBox {
                background: rgba(255, 255, 255, 0.05);
                color: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                padding: 4px 8px;
                min-width: 100px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(139, 92, 246, 0.3);
            }
            QPushButton {
                background: rgba(255, 255, 255, 0.05);
                color: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(139, 92, 246, 0.5);
            }
            QPushButton#primaryButton {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #6366F1,
                    stop: 1 #8B5CF6
                );
                color: white;
                border: none;
            }
            QPushButton#primaryButton:hover {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #7C7FFF,
                    stop: 1 #9F6FFF
                );
            }
        """)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components matching the old detector dialog"""
        layout = QVBoxLayout(self)
        
        # Main parameters group
        main_group = QGroupBox("Detection Parameters")
        main_layout = QFormLayout(main_group)
        
        # Frequency range
        self.freq_min_spin = QSpinBox()
        self.freq_min_spin.setRange(0, 20000)
        self.freq_min_spin.setValue(self.detector.freq_min if self.detector else 50)
        self.freq_min_spin.setSuffix(" Hz")
        main_layout.addRow("Min Frequency:", self.freq_min_spin)
        
        self.freq_max_spin = QSpinBox()
        self.freq_max_spin.setRange(0, 20000)
        self.freq_max_spin.setValue(self.detector.freq_max if self.detector else 1500)
        self.freq_max_spin.setSuffix(" Hz")
        main_layout.addRow("Max Frequency:", self.freq_max_spin)
        
        # Power threshold
        self.power_thresh_spin = QDoubleSpinBox()
        self.power_thresh_spin.setRange(0.01, 1.00)
        self.power_thresh_spin.setSingleStep(0.01)
        self.power_thresh_spin.setDecimals(2)
        self.power_thresh_spin.setValue(self.detector.power_threshold if self.detector else 0.2)
        main_layout.addRow("Power Threshold:", self.power_thresh_spin)
        
        # Peak prominence
        self.peak_prom_spin = QDoubleSpinBox()
        self.peak_prom_spin.setRange(0.001, 1.00)
        self.peak_prom_spin.setSingleStep(0.001)
        self.peak_prom_spin.setDecimals(3)
        self.peak_prom_spin.setValue(self.detector.peak_prominence if self.detector else 0.185)
        main_layout.addRow("Peak Prominence:", self.peak_prom_spin)
        
        layout.addWidget(main_group)
        
        # Gap handling group
        gap_group = QGroupBox("Gap Handling")
        gap_layout = QFormLayout(gap_group)
        
        self.max_gap_spin = QSpinBox()
        self.max_gap_spin.setRange(0, 100)
        self.max_gap_spin.setValue(self.detector.max_gap_frames if self.detector else 4)
        gap_layout.addRow("Max Gap Frames:", self.max_gap_spin)
        
        self.gap_power_spin = QDoubleSpinBox()
        self.gap_power_spin.setRange(0.1, 1.0)
        self.gap_power_spin.setSingleStep(0.1)
        self.gap_power_spin.setValue(self.detector.gap_power_factor if self.detector else 0.8)
        gap_layout.addRow("Gap Power Factor:", self.gap_power_spin)
        
        self.gap_prom_spin = QDoubleSpinBox()
        self.gap_prom_spin.setRange(0.1, 1.0)
        self.gap_prom_spin.setSingleStep(0.1)
        self.gap_prom_spin.setValue(self.detector.gap_prominence_factor if self.detector else 0.8)
        gap_layout.addRow("Gap Prominence Factor:", self.gap_prom_spin)
        
        layout.addWidget(gap_group)
        
        # Frequency jump group
        jump_group = QGroupBox("Frequency Jump")
        jump_layout = QFormLayout(jump_group)
        
        self.max_jump_spin = QDoubleSpinBox()
        self.max_jump_spin.setRange(0.1, 1000.0)
        self.max_jump_spin.setSingleStep(1.0)
        self.max_jump_spin.setValue(self.detector.max_freq_jump_hz if self.detector else 15.0)
        self.max_jump_spin.setSuffix(" Hz")
        jump_layout.addRow("Max Freq Jump:", self.max_jump_spin)
        
        self.gap_max_jump_spin = QDoubleSpinBox()
        self.gap_max_jump_spin.setRange(0.1, 1000.0)
        self.gap_max_jump_spin.setSingleStep(1.0)
        self.gap_max_jump_spin.setValue(self.detector.gap_max_jump_hz if self.detector else 10.0)
        self.gap_max_jump_spin.setSuffix(" Hz")
        jump_layout.addRow("Gap Max Jump:", self.gap_max_jump_spin)
        
        layout.addWidget(jump_group)
        
        # Track filtering group
        track_group = QGroupBox("Track Filtering")
        track_layout = QFormLayout(track_group)
        
        self.max_peaks_spin = QSpinBox()
        self.max_peaks_spin.setRange(1, 100)
        self.max_peaks_spin.setValue(self.detector.max_peaks_per_frame if self.detector else 20)
        track_layout.addRow("Max Peaks/Frame:", self.max_peaks_spin)
        
        self.min_length_spin = QSpinBox()
        self.min_length_spin.setRange(1, 1000)
        self.min_length_spin.setValue(self.detector.min_track_length_frames if self.detector else 14)
        track_layout.addRow("Min Track Length [frames]:", self.min_length_spin)
        
        self.min_avg_power_spin = QDoubleSpinBox()
        self.min_avg_power_spin.setRange(0.01, 1.00)
        self.min_avg_power_spin.setSingleStep(0.01)
        self.min_avg_power_spin.setDecimals(2)
        self.min_avg_power_spin.setValue(self.detector.min_track_avg_power if self.detector else 0.1)
        track_layout.addRow("Min Track Avg Power:", self.min_avg_power_spin)
        
        self.max_std_spin = QDoubleSpinBox()
        self.max_std_spin.setRange(0.1, 1000.0)
        self.max_std_spin.setSingleStep(1.0)
        self.max_std_spin.setValue(self.detector.max_track_freq_std_hz if self.detector else 70.0)
        self.max_std_spin.setSuffix(" Hz")
        track_layout.addRow("Max Track Freq Std:", self.max_std_spin)
        
        layout.addWidget(track_group)
        
        # Merge parameters group
        merge_group = QGroupBox("Track Merging")
        merge_layout = QFormLayout(merge_group)
        
        self.merge_gap_spin = QSpinBox()
        self.merge_gap_spin.setRange(0, 500)
        self.merge_gap_spin.setValue(self.detector.merge_gap_frames if self.detector else 100)
        merge_layout.addRow("Merge Gap Frames:", self.merge_gap_spin)
        
        self.merge_freq_diff_spin = QDoubleSpinBox()
        self.merge_freq_diff_spin.setRange(0.1, 1000.0)
        self.merge_freq_diff_spin.setSingleStep(1.0)
        self.merge_freq_diff_spin.setValue(self.detector.merge_max_freq_diff_hz if self.detector else 30.0)
        self.merge_freq_diff_spin.setSuffix(" Hz")
        merge_layout.addRow("Merge Max Freq Diff:", self.merge_freq_diff_spin)
        
        layout.addWidget(merge_group)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        
        # Style the buttons
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        ok_button.setObjectName("primaryButton")
        
        layout.addWidget(buttons)
        
    def accept(self):
        """Update detector parameters when OK is clicked"""
        if self.detector:
            self.detector.freq_min = self.freq_min_spin.value()
            self.detector.freq_max = self.freq_max_spin.value()
            self.detector.power_threshold = self.power_thresh_spin.value()
            self.detector.peak_prominence = self.peak_prom_spin.value()
            self.detector.max_gap_frames = self.max_gap_spin.value()
            self.detector.gap_power_factor = self.gap_power_spin.value()
            self.detector.gap_prominence_factor = self.gap_prom_spin.value()
            self.detector.max_freq_jump_hz = self.max_jump_spin.value()
            self.detector.gap_max_jump_hz = self.gap_max_jump_spin.value()
            self.detector.max_peaks_per_frame = self.max_peaks_spin.value()
            self.detector.min_track_length_frames = self.min_length_spin.value()
            self.detector.min_track_avg_power = self.min_avg_power_spin.value()
            self.detector.max_track_freq_std_hz = self.max_std_spin.value()
            self.detector.merge_gap_frames = self.merge_gap_spin.value()
            self.detector.merge_max_freq_diff_hz = self.merge_freq_diff_spin.value()
            self.detector.detection_method = "peaks"
            
            # Emit signal with parameters
            params = {
                'freq_min': self.detector.freq_min,
                'freq_max': self.detector.freq_max,
                'power_threshold': self.detector.power_threshold,
                'peak_prominence': self.detector.peak_prominence,
                'max_gap_frames': self.detector.max_gap_frames,
                'gap_power_factor': self.detector.gap_power_factor,
                'gap_prominence_factor': self.detector.gap_prominence_factor,
                'max_freq_jump_hz': self.detector.max_freq_jump_hz,
                'gap_max_jump_hz': self.detector.gap_max_jump_hz,
                'max_peaks_per_frame': self.detector.max_peaks_per_frame,
                'min_track_length_frames': self.detector.min_track_length_frames,
                'min_track_avg_power': self.detector.min_track_avg_power,
                'max_track_freq_std_hz': self.detector.max_track_freq_std_hz,
                'merge_gap_frames': self.detector.merge_gap_frames,
                'merge_max_freq_diff_hz': self.detector.merge_max_freq_diff_hz,
                'detection_method': self.detector.detection_method
            }
            self.detection_started.emit(params)
            
        super().accept()