"""
Parameter dialog for the Physical Doppler Detector.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QPushButton,
    QGroupBox, QDialogButtonBox, QTabWidget, QWidget
)
from PyQt5.QtCore import Qt


class PhysicalDetectorParamsDialog(QDialog):
    """Parameter dialog for the physics-based Doppler detector."""

    def __init__(self, parent=None, detector=None):
        super().__init__(parent)
        self.setWindowTitle("Physical Doppler Detector Parameters")
        self.detector = detector
        self.setModal(True)
        self.resize(500, 600)

        # Main layout
        layout = QVBoxLayout(self)

        # Create tabs for different parameter groups
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Tab 1: Aircraft/Drone Parameters
        aircraft_tab = QWidget()
        aircraft_layout = QGridLayout(aircraft_tab)

        aircraft_layout.addWidget(QLabel("Min Altitude (m):"), 0, 0)
        self.min_altitude_spin = QDoubleSpinBox()
        self.min_altitude_spin.setRange(10, 5000)
        self.min_altitude_spin.setValue(detector.min_altitude_m)
        aircraft_layout.addWidget(self.min_altitude_spin, 0, 1)

        aircraft_layout.addWidget(QLabel("Max Altitude (m):"), 1, 0)
        self.max_altitude_spin = QDoubleSpinBox()
        self.max_altitude_spin.setRange(10, 10000)
        self.max_altitude_spin.setValue(detector.max_altitude_m)
        aircraft_layout.addWidget(self.max_altitude_spin, 1, 1)

        aircraft_layout.addWidget(QLabel("Min Speed (m/s):"), 2, 0)
        self.min_speed_spin = QDoubleSpinBox()
        self.min_speed_spin.setRange(1, 100)
        self.min_speed_spin.setValue(detector.min_speed_ms)
        aircraft_layout.addWidget(self.min_speed_spin, 2, 1)

        aircraft_layout.addWidget(QLabel("Max Speed (m/s):"), 3, 0)
        self.max_speed_spin = QDoubleSpinBox()
        self.max_speed_spin.setRange(5, 500)
        self.max_speed_spin.setValue(detector.max_speed_ms)
        aircraft_layout.addWidget(self.max_speed_spin, 3, 1)

        aircraft_layout.setRowStretch(4, 1)
        tabs.addTab(aircraft_tab, "Aircraft/Drone")

        # Tab 2: BPF Detection Parameters
        bpf_tab = QWidget()
        bpf_layout = QGridLayout(bpf_tab)

        bpf_layout.addWidget(QLabel("Min BPF (Hz):"), 0, 0)
        self.min_bpf_spin = QDoubleSpinBox()
        self.min_bpf_spin.setRange(10, 1000)
        self.min_bpf_spin.setValue(detector.min_bpf_hz)
        bpf_layout.addWidget(self.min_bpf_spin, 0, 1)

        bpf_layout.addWidget(QLabel("Max BPF (Hz):"), 1, 0)
        self.max_bpf_spin = QDoubleSpinBox()
        self.max_bpf_spin.setRange(50, 2000)
        self.max_bpf_spin.setValue(detector.max_bpf_hz)
        bpf_layout.addWidget(self.max_bpf_spin, 1, 1)

        bpf_layout.addWidget(QLabel("BPF Tolerance (Hz):"), 2, 0)
        self.bpf_tolerance_spin = QDoubleSpinBox()
        self.bpf_tolerance_spin.setRange(1, 50)
        self.bpf_tolerance_spin.setValue(detector.bpf_tolerance_hz)
        bpf_layout.addWidget(self.bpf_tolerance_spin, 2, 1)

        bpf_layout.setRowStretch(3, 1)
        tabs.addTab(bpf_tab, "BPF Detection")

        # Tab 3: Event Characteristics
        event_tab = QWidget()
        event_layout = QGridLayout(event_tab)

        event_layout.addWidget(QLabel("Min Event Duration (s):"), 0, 0)
        self.min_duration_spin = QDoubleSpinBox()
        self.min_duration_spin.setRange(1, 60)
        self.min_duration_spin.setValue(detector.min_event_duration_s)
        event_layout.addWidget(self.min_duration_spin, 0, 1)

        event_layout.addWidget(QLabel("Max Event Duration (s):"), 1, 0)
        self.max_duration_spin = QDoubleSpinBox()
        self.max_duration_spin.setRange(10, 300)
        self.max_duration_spin.setValue(detector.max_event_duration_s)
        event_layout.addWidget(self.max_duration_spin, 1, 1)

        event_layout.addWidget(QLabel("Min SNR (dB):"), 2, 0)
        self.min_snr_spin = QDoubleSpinBox()
        self.min_snr_spin.setRange(0, 30)
        self.min_snr_spin.setValue(detector.min_snr_db)
        event_layout.addWidget(self.min_snr_spin, 2, 1)

        event_layout.setRowStretch(3, 1)
        tabs.addTab(event_tab, "Event Validation")

        # Tab 4: Doppler Validation
        doppler_tab = QWidget()
        doppler_layout = QGridLayout(doppler_tab)

        doppler_layout.addWidget(QLabel("Max Doppler Rate (Hz/s):"), 0, 0)
        self.max_doppler_rate_spin = QDoubleSpinBox()
        self.max_doppler_rate_spin.setRange(0.1, 50)
        self.max_doppler_rate_spin.setValue(detector.max_doppler_rate_hz_s)
        doppler_layout.addWidget(self.max_doppler_rate_spin, 0, 1)

        doppler_layout.addWidget(QLabel("Min Doppler Shift (Hz):"), 1, 0)
        self.min_doppler_shift_spin = QDoubleSpinBox()
        self.min_doppler_shift_spin.setRange(0.5, 20)
        self.min_doppler_shift_spin.setValue(detector.min_doppler_shift_hz)
        doppler_layout.addWidget(self.min_doppler_shift_spin, 1, 1)

        doppler_layout.addWidget(QLabel("Peak Prominence Factor:"), 2, 0)
        self.peak_prominence_spin = QDoubleSpinBox()
        self.peak_prominence_spin.setRange(1.1, 5.0)
        self.peak_prominence_spin.setDecimals(1)
        self.peak_prominence_spin.setValue(detector.peak_prominence_factor)
        doppler_layout.addWidget(self.peak_prominence_spin, 2, 1)

        doppler_layout.setRowStretch(3, 1)
        tabs.addTab(doppler_tab, "Doppler Validation")

        # Preset buttons
        preset_group = QGroupBox("Presets")
        preset_layout = QHBoxLayout(preset_group)
        
        drone_btn = QPushButton("Small Drone")
        drone_btn.clicked.connect(self.apply_drone_preset)
        preset_layout.addWidget(drone_btn)
        
        aircraft_btn = QPushButton("Light Aircraft")
        aircraft_btn.clicked.connect(self.apply_aircraft_preset)
        preset_layout.addWidget(aircraft_btn)
        
        heli_btn = QPushButton("Helicopter")
        heli_btn.clicked.connect(self.apply_helicopter_preset)
        preset_layout.addWidget(heli_btn)
        
        layout.addWidget(preset_group)

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def apply_drone_preset(self):
        """Apply settings optimized for small drones."""
        self.min_altitude_spin.setValue(20.0)
        self.max_altitude_spin.setValue(300.0)
        self.min_speed_spin.setValue(3.0)
        self.max_speed_spin.setValue(30.0)
        self.min_bpf_spin.setValue(100.0)
        self.max_bpf_spin.setValue(400.0)
        self.bpf_tolerance_spin.setValue(20.0)
        self.min_duration_spin.setValue(5.0)
        self.max_duration_spin.setValue(120.0)
        self.min_snr_spin.setValue(5.0)
        self.max_doppler_rate_spin.setValue(10.0)
        self.min_doppler_shift_spin.setValue(3.0)

    def apply_aircraft_preset(self):
        """Apply settings optimized for light aircraft."""
        self.min_altitude_spin.setValue(100.0)
        self.max_altitude_spin.setValue(2000.0)
        self.min_speed_spin.setValue(20.0)
        self.max_speed_spin.setValue(100.0)
        self.min_bpf_spin.setValue(50.0)
        self.max_bpf_spin.setValue(200.0)
        self.bpf_tolerance_spin.setValue(15.0)
        self.min_duration_spin.setValue(10.0)
        self.max_duration_spin.setValue(180.0)
        self.min_snr_spin.setValue(6.0)
        self.max_doppler_rate_spin.setValue(5.0)
        self.min_doppler_shift_spin.setValue(2.0)

    def apply_helicopter_preset(self):
        """Apply settings optimized for helicopters."""
        self.min_altitude_spin.setValue(50.0)
        self.max_altitude_spin.setValue(1000.0)
        self.min_speed_spin.setValue(5.0)
        self.max_speed_spin.setValue(70.0)
        self.min_bpf_spin.setValue(15.0)
        self.max_bpf_spin.setValue(100.0)
        self.bpf_tolerance_spin.setValue(10.0)
        self.min_duration_spin.setValue(8.0)
        self.max_duration_spin.setValue(150.0)
        self.min_snr_spin.setValue(7.0)
        self.max_doppler_rate_spin.setValue(3.0)
        self.min_doppler_shift_spin.setValue(1.5)

    def accept(self):
        """Apply parameters to detector and close dialog."""
        d = self.detector
        
        # Aircraft/Drone parameters
        d.min_altitude_m = self.min_altitude_spin.value()
        d.max_altitude_m = self.max_altitude_spin.value()
        d.min_speed_ms = self.min_speed_spin.value()
        d.max_speed_ms = self.max_speed_spin.value()
        
        # BPF parameters
        d.min_bpf_hz = self.min_bpf_spin.value()
        d.max_bpf_hz = self.max_bpf_spin.value()
        d.bpf_tolerance_hz = self.bpf_tolerance_spin.value()
        
        # Event parameters
        d.min_event_duration_s = self.min_duration_spin.value()
        d.max_event_duration_s = self.max_duration_spin.value()
        d.min_snr_db = self.min_snr_spin.value()
        
        # Doppler parameters
        d.max_doppler_rate_hz_s = self.max_doppler_rate_spin.value()
        d.min_doppler_shift_hz = self.min_doppler_shift_spin.value()
        d.peak_prominence_factor = self.peak_prominence_spin.value()
        
        super().accept()