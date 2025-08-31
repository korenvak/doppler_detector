from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QSpinBox, QComboBox,
    QDialogButtonBox, QLabel, QPushButton, QHBoxLayout,
    QVBoxLayout, QGroupBox, QCheckBox
)
from PyQt5.QtCore import pyqtSignal


class SpectrogramSettingsDialog(QDialog):
    """Enhanced dialog to tweak spectrogram display parameters."""
    
    # Signal emitted when Apply is clicked
    paramsChanged = pyqtSignal(dict)

    def __init__(self, parent=None, params=None):
        super().__init__(parent)
        self.setWindowTitle("Spectrogram Settings")
        self.parent_window = parent
        
        main_layout = QVBoxLayout(self)
        
        # FFT Settings Group
        fft_group = QGroupBox("FFT Settings")
        fft_layout = QFormLayout()
        
        p = params or {}

        # Window size with power of 2 options
        self.nperseg_spin = QSpinBox()
        self.nperseg_spin.setRange(256, 16384)
        self.nperseg_spin.setSingleStep(256)
        self.nperseg_spin.setValue(p.get("window_size", 4096))
        self.nperseg_spin.setToolTip("FFT window size (samples). Higher values = better frequency resolution, worse time resolution")
        
        window_layout = QHBoxLayout()
        window_layout.addWidget(self.nperseg_spin)
        
        # Quick select buttons for common window sizes
        for size in [1024, 2048, 4096, 8192]:
            btn = QPushButton(str(size))
            btn.setMaximumWidth(60)
            btn.clicked.connect(lambda checked, s=size: self.nperseg_spin.setValue(s))
            window_layout.addWidget(btn)
        
        fft_layout.addRow("Window Size:", window_layout)
        
        # Overlap
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 95)
        self.overlap_spin.setSuffix(" %")
        self.overlap_spin.setValue(p.get("overlap", 75))
        self.overlap_spin.setToolTip("Overlap percentage between consecutive windows")
        
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(self.overlap_spin)
        
        # Quick select buttons for common overlaps
        for overlap in [50, 75, 90]:
            btn = QPushButton(f"{overlap}%")
            btn.setMaximumWidth(60)
            btn.clicked.connect(lambda checked, o=overlap: self.overlap_spin.setValue(o))
            overlap_layout.addWidget(btn)
        
        fft_layout.addRow("Overlap:", overlap_layout)
        
        # Window function
        self.window_combo = QComboBox()
        self.window_combo.addItems(["blackmanharris", "hann", "hamming", "blackman", "bartlett", "flattop", "tukey"])
        self.window_combo.setCurrentText(p.get("window", "blackmanharris"))
        self.window_combo.setToolTip("Window function to apply before FFT")
        fft_layout.addRow("Window Function:", self.window_combo)
        
        fft_group.setLayout(fft_layout)
        main_layout.addWidget(fft_group)
        
        # Display Settings Group
        display_group = QGroupBox("Display Settings")
        display_layout = QFormLayout()
        
        # Colormap
        self.cmap_combo = QComboBox()
        colormaps = ["gray", "viridis", "magma", "inferno", "plasma", "hot", "cool", "jet", "turbo", "twilight"]
        self.cmap_combo.addItems(colormaps)
        self.cmap_combo.setCurrentText(p.get("colormap", "magma"))
        self.cmap_combo.setToolTip("Color scheme for spectrogram display")
        display_layout.addRow("Colormap:", self.cmap_combo)
        
        # Frequency range
        freq_range_layout = QHBoxLayout()
        
        self.freq_min_spin = QSpinBox()
        self.freq_min_spin.setRange(0, 20000)
        self.freq_min_spin.setSuffix(" Hz")
        self.freq_min_spin.setValue(p.get("freq_min", 0))
        
        self.freq_max_spin = QSpinBox()
        self.freq_max_spin.setRange(100, 20000)
        self.freq_max_spin.setSuffix(" Hz")
        self.freq_max_spin.setValue(p.get("freq_max", 8000))
        
        freq_range_layout.addWidget(QLabel("Min:"))
        freq_range_layout.addWidget(self.freq_min_spin)
        freq_range_layout.addWidget(QLabel("Max:"))
        freq_range_layout.addWidget(self.freq_max_spin)
        
        display_layout.addRow("Frequency Range:", freq_range_layout)
        
        # Dynamic range
        self.dynamic_range_spin = QSpinBox()
        self.dynamic_range_spin.setRange(20, 120)
        self.dynamic_range_spin.setSuffix(" dB")
        self.dynamic_range_spin.setValue(p.get("dynamic_range", 80))
        self.dynamic_range_spin.setToolTip("Dynamic range for display (in dB)")
        display_layout.addRow("Dynamic Range:", self.dynamic_range_spin)
        
        # Log scale option
        self.log_scale_check = QCheckBox("Logarithmic frequency scale")
        self.log_scale_check.setChecked(p.get("log_scale", False))
        display_layout.addRow(self.log_scale_check)
        
        display_group.setLayout(display_layout)
        main_layout.addWidget(display_group)
        
        # Processing Settings Group
        processing_group = QGroupBox("Processing Settings")
        processing_layout = QFormLayout()
        
        # Smoothing
        self.smooth_sigma_spin = QSpinBox()
        self.smooth_sigma_spin.setRange(0, 10)
        self.smooth_sigma_spin.setValue(int(p.get("smooth_sigma", 1.5) * 2))
        self.smooth_sigma_spin.setToolTip("Gaussian smoothing sigma (0 = no smoothing)")
        processing_layout.addRow("Smoothing:", self.smooth_sigma_spin)
        
        # Median filter
        self.median_filter_check = QCheckBox("Apply median filter")
        self.median_filter_check.setChecked(p.get("median_filter", True))
        processing_layout.addRow(self.median_filter_check)
        
        processing_group.setLayout(processing_layout)
        main_layout.addWidget(processing_group)
        
        # Info label
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("QLabel { color: #888; font-style: italic; }")
        main_layout.addWidget(self.info_label)
        self._update_info()
        
        # Connect signals to update info
        self.nperseg_spin.valueChanged.connect(self._update_info)
        self.overlap_spin.valueChanged.connect(self._update_info)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_settings)
        self.apply_btn.setToolTip("Apply settings without closing dialog")
        
        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self.reset_defaults)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        
        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(buttons)
        
        main_layout.addLayout(button_layout)
        
        self.setMinimumWidth(500)

    def get_params(self):
        return {
            "window_size": self.nperseg_spin.value(),
            "overlap": self.overlap_spin.value(),
            "colormap": self.cmap_combo.currentText(),
            "window": self.window_combo.currentText(),
            "freq_min": self.freq_min_spin.value(),
            "freq_max": self.freq_max_spin.value(),
            "dynamic_range": self.dynamic_range_spin.value(),
            "log_scale": self.log_scale_check.isChecked(),
            "smooth_sigma": self.smooth_sigma_spin.value() / 2.0,
            "median_filter": self.median_filter_check.isChecked(),
            "median_filter_size": (3, 1) if self.median_filter_check.isChecked() else (1, 1)
        }
    
    def apply_settings(self):
        """Apply settings without closing the dialog."""
        params = self.get_params()
        self.paramsChanged.emit(params)
        
        # If parent window has the method, call it directly
        if hasattr(self.parent_window, 'spectrogram_params'):
            self.parent_window.spectrogram_params.update(params)
            if hasattr(self.parent_window, 'current_file') and self.parent_window.current_file:
                if hasattr(self.parent_window, 'load_file_from_path'):
                    self.parent_window.load_file_from_path(
                        self.parent_window.current_file, 
                        maintain_view=True
                    )
    
    def reset_defaults(self):
        """Reset all settings to default values."""
        self.nperseg_spin.setValue(4096)
        self.overlap_spin.setValue(75)
        self.window_combo.setCurrentText("blackmanharris")
        self.cmap_combo.setCurrentText("magma")
        self.freq_min_spin.setValue(0)
        self.freq_max_spin.setValue(8000)
        self.dynamic_range_spin.setValue(80)
        self.log_scale_check.setChecked(False)
        self.smooth_sigma_spin.setValue(3)  # 1.5 * 2
        self.median_filter_check.setChecked(True)
        self._update_info()
    
    def _update_info(self):
        """Update the info label with calculated parameters."""
        window_size = self.nperseg_spin.value()
        overlap = self.overlap_spin.value()
        
        # Calculate time and frequency resolution (approximate)
        # Assuming 44.1 kHz sample rate for display purposes
        sample_rate = 44100
        time_res = (window_size * (1 - overlap/100)) / sample_rate * 1000  # ms
        freq_res = sample_rate / window_size  # Hz
        
        self.info_label.setText(
            f"Time resolution: ~{time_res:.1f} ms, "
            f"Frequency resolution: ~{freq_res:.1f} Hz"
        )
