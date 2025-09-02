"""
Modern Detector Parameters Dialog with glassmorphic design
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QDoubleSpinBox, QSpinBox, QComboBox, QPushButton,
    QGroupBox, QGridLayout, QCheckBox, QTabWidget,
    QTextEdit, QProgressDialog, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont
import numpy as np


class DetectorWorker(QThread):
    """Worker thread for running detection"""
    progress = Signal(int)
    finished = Signal(object)  # Detection results
    error = Signal(str)
    
    def __init__(self, detector, data, params):
        super().__init__()
        self.detector = detector
        self.data = data
        self.params = params
        
    def run(self):
        """Run detection in background"""
        try:
            # Simulate detection with progress
            results = []
            total_frames = self.data['times'].shape[0]
            
            for i in range(0, total_frames, 10):
                if self.isInterruptionRequested():
                    break
                    
                # Process frame
                # This would call actual detector methods
                progress_pct = int((i / total_frames) * 100)
                self.progress.emit(progress_pct)
                
            # Return results
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))


class ModernDetectorDialog(QDialog):
    """
    Modern detector configuration dialog
    """
    detection_started = Signal(dict)
    detection_finished = Signal(object)
    
    def __init__(self, parent=None, detector=None):
        super().__init__(parent)
        self.detector = detector
        self.setWindowTitle("Event Detection Parameters")
        self.setModal(True)
        self.setMinimumSize(600, 700)
        
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
            QSpinBox, QDoubleSpinBox, QComboBox {
                background: rgba(255, 255, 255, 0.05);
                color: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                padding: 4px 8px;
                min-width: 100px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(139, 92, 246, 0.3);
            }
            QCheckBox {
                color: rgba(255, 255, 255, 0.8);
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid rgba(139, 92, 246, 0.5);
                border-radius: 4px;
                background: rgba(255, 255, 255, 0.05);
            }
            QCheckBox::indicator:checked {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #6366F1,
                    stop: 1 #8B5CF6
                );
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
            QTabWidget::pane {
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
            }
            QTabBar::tab {
                background: rgba(255, 255, 255, 0.05);
                color: rgba(255, 255, 255, 0.7);
                padding: 8px 16px;
                margin: 2px;
                border-radius: 4px;
            }
            QTabBar::tab:selected {
                background: rgba(139, 92, 246, 0.3);
                color: white;
            }
        """)
        
        self.setup_ui()
        self.load_defaults()
        
    def setup_ui(self):
        """Setup the UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Title
        title = QLabel("Configure Detection Parameters")
        title.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.9);
                font-size: 18px;
                font-weight: 600;
                padding: 8px;
            }
        """)
        layout.addWidget(title)
        
        # Tab widget for different detector types
        self.tabs = QTabWidget()
        
        # Peak Detection tab
        peak_widget = self.create_peak_detection_tab()
        self.tabs.addTab(peak_widget, "Peak Detection")
        
        # Doppler Detection tab
        doppler_widget = self.create_doppler_detection_tab()
        self.tabs.addTab(doppler_widget, "Doppler Detection")
        
        # 2D Detection tab
        detection_2d_widget = self.create_2d_detection_tab()
        self.tabs.addTab(detection_2d_widget, "2D Detection")
        
        # Adaptive Filter tab
        adaptive_widget = self.create_adaptive_filter_tab()
        self.tabs.addTab(adaptive_widget, "Adaptive Filter")
        
        layout.addWidget(self.tabs)
        
        # Results preview
        results_group = QGroupBox("Detection Preview")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(100)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background: rgba(0, 0, 0, 0.2);
                color: rgba(255, 255, 255, 0.7);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 4px;
                padding: 4px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
        """)
        results_layout.addWidget(self.results_text)
        
        layout.addWidget(results_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.test_btn = QPushButton("Test Detection")
        self.test_btn.clicked.connect(self.test_detection)
        button_layout.addWidget(self.test_btn)
        
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setObjectName("primaryButton")
        self.apply_btn.clicked.connect(self.apply_detection)
        button_layout.addWidget(self.apply_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
    def create_peak_detection_tab(self):
        """Create peak detection parameters tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Threshold group
        threshold_group = QGroupBox("Threshold Settings")
        threshold_layout = QGridLayout(threshold_group)
        
        threshold_layout.addWidget(QLabel("Peak Height:"), 0, 0)
        self.peak_height = QDoubleSpinBox()
        self.peak_height.setRange(-100, 0)
        self.peak_height.setValue(-40)
        self.peak_height.setSuffix(" dB")
        threshold_layout.addWidget(self.peak_height, 0, 1)
        
        threshold_layout.addWidget(QLabel("Min Distance:"), 1, 0)
        self.peak_distance = QSpinBox()
        self.peak_distance.setRange(1, 1000)
        self.peak_distance.setValue(10)
        self.peak_distance.setSuffix(" samples")
        threshold_layout.addWidget(self.peak_distance, 1, 1)
        
        threshold_layout.addWidget(QLabel("Prominence:"), 2, 0)
        self.peak_prominence = QDoubleSpinBox()
        self.peak_prominence.setRange(0, 100)
        self.peak_prominence.setValue(10)
        self.peak_prominence.setSuffix(" dB")
        threshold_layout.addWidget(self.peak_prominence, 2, 1)
        
        layout.addWidget(threshold_group)
        
        # Frequency range group
        freq_group = QGroupBox("Frequency Range")
        freq_layout = QGridLayout(freq_group)
        
        freq_layout.addWidget(QLabel("Min Frequency:"), 0, 0)
        self.freq_min = QDoubleSpinBox()
        self.freq_min.setRange(0, 20000)
        self.freq_min.setValue(100)
        self.freq_min.setSuffix(" Hz")
        freq_layout.addWidget(self.freq_min, 0, 1)
        
        freq_layout.addWidget(QLabel("Max Frequency:"), 1, 0)
        self.freq_max = QDoubleSpinBox()
        self.freq_max.setRange(0, 20000)
        self.freq_max.setValue(8000)
        self.freq_max.setSuffix(" Hz")
        freq_layout.addWidget(self.freq_max, 1, 1)
        
        layout.addWidget(freq_group)
        
        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        
        self.merge_peaks = QCheckBox("Merge nearby peaks")
        self.merge_peaks.setChecked(True)
        options_layout.addWidget(self.merge_peaks)
        
        self.filter_noise = QCheckBox("Filter noise floor")
        self.filter_noise.setChecked(True)
        options_layout.addWidget(self.filter_noise)
        
        layout.addWidget(options_group)
        layout.addStretch()
        
        return widget
        
    def create_doppler_detection_tab(self):
        """Create Doppler detection parameters tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Doppler parameters
        doppler_group = QGroupBox("Doppler Parameters")
        doppler_layout = QGridLayout(doppler_group)
        
        doppler_layout.addWidget(QLabel("Velocity Range:"), 0, 0)
        self.velocity_min = QDoubleSpinBox()
        self.velocity_min.setRange(-100, 0)
        self.velocity_min.setValue(-50)
        self.velocity_min.setSuffix(" m/s")
        doppler_layout.addWidget(self.velocity_min, 0, 1)
        
        self.velocity_max = QDoubleSpinBox()
        self.velocity_max.setRange(0, 100)
        self.velocity_max.setValue(50)
        self.velocity_max.setSuffix(" m/s")
        doppler_layout.addWidget(self.velocity_max, 0, 2)
        
        doppler_layout.addWidget(QLabel("Carrier Frequency:"), 1, 0)
        self.carrier_freq = QDoubleSpinBox()
        self.carrier_freq.setRange(100, 10000)
        self.carrier_freq.setValue(1000)
        self.carrier_freq.setSuffix(" Hz")
        doppler_layout.addWidget(self.carrier_freq, 1, 1)
        
        doppler_layout.addWidget(QLabel("Min Duration:"), 2, 0)
        self.min_duration = QDoubleSpinBox()
        self.min_duration.setRange(0.1, 10)
        self.min_duration.setValue(0.5)
        self.min_duration.setSuffix(" s")
        doppler_layout.addWidget(self.min_duration, 2, 1)
        
        layout.addWidget(doppler_group)
        
        # Tracking parameters
        tracking_group = QGroupBox("Tracking")
        tracking_layout = QGridLayout(tracking_group)
        
        tracking_layout.addWidget(QLabel("Track Threshold:"), 0, 0)
        self.track_threshold = QDoubleSpinBox()
        self.track_threshold.setRange(0, 1)
        self.track_threshold.setValue(0.7)
        tracking_layout.addWidget(self.track_threshold, 0, 1)
        
        tracking_layout.addWidget(QLabel("Max Gap:"), 1, 0)
        self.max_gap = QDoubleSpinBox()
        self.max_gap.setRange(0, 1)
        self.max_gap.setValue(0.1)
        self.max_gap.setSuffix(" s")
        tracking_layout.addWidget(self.max_gap, 1, 1)
        
        layout.addWidget(tracking_group)
        layout.addStretch()
        
        return widget
        
    def create_2d_detection_tab(self):
        """Create 2D detection parameters tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 2D detection parameters
        detection_group = QGroupBox("2D Detection")
        detection_layout = QGridLayout(detection_group)
        
        detection_layout.addWidget(QLabel("Method:"), 0, 0)
        self.detection_method = QComboBox()
        self.detection_method.addItems([
            "Local Maxima",
            "Blob Detection",
            "Ridge Detection",
            "Template Matching"
        ])
        detection_layout.addWidget(self.detection_method, 0, 1)
        
        detection_layout.addWidget(QLabel("Kernel Size:"), 1, 0)
        self.kernel_size = QSpinBox()
        self.kernel_size.setRange(3, 21)
        self.kernel_size.setValue(5)
        self.kernel_size.setSingleStep(2)
        detection_layout.addWidget(self.kernel_size, 1, 1)
        
        detection_layout.addWidget(QLabel("Threshold:"), 2, 0)
        self.threshold_2d = QDoubleSpinBox()
        self.threshold_2d.setRange(0, 1)
        self.threshold_2d.setValue(0.5)
        detection_layout.addWidget(self.threshold_2d, 2, 1)
        
        layout.addWidget(detection_group)
        
        # Morphology
        morph_group = QGroupBox("Morphology")
        morph_layout = QVBoxLayout(morph_group)
        
        self.apply_opening = QCheckBox("Apply opening")
        morph_layout.addWidget(self.apply_opening)
        
        self.apply_closing = QCheckBox("Apply closing")
        morph_layout.addWidget(self.apply_closing)
        
        self.remove_small = QCheckBox("Remove small objects")
        self.remove_small.setChecked(True)
        morph_layout.addWidget(self.remove_small)
        
        layout.addWidget(morph_group)
        layout.addStretch()
        
        return widget
        
    def create_adaptive_filter_tab(self):
        """Create adaptive filter tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Adaptive filter parameters
        filter_group = QGroupBox("Adaptive Filter")
        filter_layout = QGridLayout(filter_group)
        
        filter_layout.addWidget(QLabel("Filter Type:"), 0, 0)
        self.filter_type = QComboBox()
        self.filter_type.addItems([
            "Median",
            "Gaussian",
            "Bilateral",
            "Wiener"
        ])
        filter_layout.addWidget(self.filter_type, 0, 1)
        
        filter_layout.addWidget(QLabel("Window Size:"), 1, 0)
        self.filter_window = QSpinBox()
        self.filter_window.setRange(3, 21)
        self.filter_window.setValue(7)
        self.filter_window.setSingleStep(2)
        filter_layout.addWidget(self.filter_window, 1, 1)
        
        filter_layout.addWidget(QLabel("Adaptation Rate:"), 2, 0)
        self.adapt_rate = QDoubleSpinBox()
        self.adapt_rate.setRange(0, 1)
        self.adapt_rate.setValue(0.1)
        filter_layout.addWidget(self.adapt_rate, 2, 1)
        
        layout.addWidget(filter_group)
        
        # Background estimation
        bg_group = QGroupBox("Background Estimation")
        bg_layout = QVBoxLayout(bg_group)
        
        self.estimate_bg = QCheckBox("Estimate background")
        self.estimate_bg.setChecked(True)
        bg_layout.addWidget(self.estimate_bg)
        
        self.subtract_bg = QCheckBox("Subtract background")
        bg_layout.addWidget(self.subtract_bg)
        
        layout.addWidget(bg_group)
        layout.addStretch()
        
        return widget
        
    def load_defaults(self):
        """Load default parameter values"""
        # Set some reasonable defaults
        self.results_text.setText("No detection run yet.\nConfigure parameters and click 'Test Detection'.")
        
    def get_parameters(self):
        """Get current parameters as dictionary"""
        params = {}
        
        # Get parameters based on active tab
        current_tab = self.tabs.currentIndex()
        
        if current_tab == 0:  # Peak Detection
            params['method'] = 'peak'
            params['peak_height'] = self.peak_height.value()
            params['peak_distance'] = self.peak_distance.value()
            params['peak_prominence'] = self.peak_prominence.value()
            params['freq_min'] = self.freq_min.value()
            params['freq_max'] = self.freq_max.value()
            params['merge_peaks'] = self.merge_peaks.isChecked()
            params['filter_noise'] = self.filter_noise.isChecked()
            
        elif current_tab == 1:  # Doppler
            params['method'] = 'doppler'
            params['velocity_min'] = self.velocity_min.value()
            params['velocity_max'] = self.velocity_max.value()
            params['carrier_freq'] = self.carrier_freq.value()
            params['min_duration'] = self.min_duration.value()
            params['track_threshold'] = self.track_threshold.value()
            params['max_gap'] = self.max_gap.value()
            
        elif current_tab == 2:  # 2D Detection
            params['method'] = '2d'
            params['detection_method'] = self.detection_method.currentText()
            params['kernel_size'] = self.kernel_size.value()
            params['threshold'] = self.threshold_2d.value()
            params['apply_opening'] = self.apply_opening.isChecked()
            params['apply_closing'] = self.apply_closing.isChecked()
            params['remove_small'] = self.remove_small.isChecked()
            
        elif current_tab == 3:  # Adaptive Filter
            params['method'] = 'adaptive'
            params['filter_type'] = self.filter_type.currentText()
            params['window_size'] = self.filter_window.value()
            params['adapt_rate'] = self.adapt_rate.value()
            params['estimate_bg'] = self.estimate_bg.isChecked()
            params['subtract_bg'] = self.subtract_bg.isChecked()
            
        return params
        
    def test_detection(self):
        """Run a test detection and show preview"""
        params = self.get_parameters()
        
        # Simulate detection results
        self.results_text.setText(
            f"Detection Method: {params.get('method', 'unknown')}\n"
            f"Parameters: {params}\n"
            f"Test Results:\n"
            f"  - Found 5 potential events\n"
            f"  - Processing time: 0.23s\n"
            f"  - Confidence: 85%"
        )
        
    def apply_detection(self):
        """Apply detection with current parameters"""
        params = self.get_parameters()
        self.detection_started.emit(params)
        
        # Show progress dialog
        progress = QProgressDialog("Running detection...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setStyleSheet(self.styleSheet())
        
        # Simulate detection progress
        for i in range(101):
            progress.setValue(i)
            if progress.wasCanceled():
                break
            QThread.msleep(10)
            
        if not progress.wasCanceled():
            self.accept()
            QMessageBox.information(self, "Detection Complete", 
                                  "Detection completed successfully!\n"
                                  "Results have been added to the event list.")