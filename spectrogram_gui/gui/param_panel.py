from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QLabel, QGroupBox, QCheckBox, QHBoxLayout,
    QSlider, QSpinBox, QFormLayout, QComboBox
)
from PyQt5.QtCore import Qt, QPropertyAnimation


class ParamPanel(QFrame):
    """Collapsible bottom drawer showing filters, detection parameters and stats."""

    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.setObjectName("paramPanel")
        self._expanded = False
        self.setMaximumHeight(0)
        self.animation = QPropertyAnimation(self, b"maximumHeight")
        self.animation.setDuration(250)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ----- Filter Settings -----
        filter_box = QGroupBox("Filter Settings")
        f_layout = QFormLayout(filter_box)
        self.filter_checks = {}
        self.filter_params = {}
        for name in ["NLMS", "LMS", "ALE", "RLS", "Wiener"]:
            cb = QCheckBox(name)
            self.filter_checks[name] = cb
            if name == "ALE":
                row = QHBoxLayout()
                mu = QSpinBox(); mu.setRange(0, 100); mu.setValue(1)
                lam = QSpinBox(); lam.setRange(0, 100); lam.setValue(1)
                delay = QSpinBox(); delay.setRange(1, 20); delay.setValue(1)
                self.filter_params[name] = (mu, lam, delay)
                row.addWidget(cb); row.addWidget(mu); row.addWidget(lam); row.addWidget(delay)
                f_layout.addRow(row)
            else:
                val = QSpinBox(); val.setRange(0, 100); val.setValue(1)
                self.filter_params[name] = (val,)
                row = QHBoxLayout(); row.addWidget(cb); row.addWidget(val)
                f_layout.addRow(row)
        layout.addWidget(filter_box)

        # ----- Detection Parameters -----
        det_box = QGroupBox("Detection Parameters")
        d_layout = QFormLayout(det_box)
        self.threshold_slider, self.threshold_spin = self._add_slider_spin(d_layout, "Power Threshold", 0, 100)
        self.prom_slider, self.prom_spin = self._add_slider_spin(d_layout, "Peak Prominence", 0, 50)
        self.jump_slider, self.jump_spin = self._add_slider_spin(d_layout, "Max Hz Jump", 0, 400)
        self.len_slider, self.len_spin = self._add_slider_spin(d_layout, "Min Track Length", 0, 10)
        self.std_slider, self.std_spin = self._add_slider_spin(d_layout, "Track Freq Std Max", 0, 50)
        layout.addWidget(det_box)

        # ----- Spectrogram View -----
        spec_box = QGroupBox("Spectrogram View")
        s_layout = QFormLayout(spec_box)
        self.range_slider, self.range_spin = self._add_slider_spin(s_layout, "Dynamic Range (dB)", 40, 120)
        self.fft_combo = QComboBox()
        self.fft_combo.addItems(["1024", "2048", "4096", "8192"])
        s_layout.addRow("FFT Window", self.fft_combo)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["gray", "viridis", "magma", "inferno", "plasma"])
        self.cmap_combo.currentTextChanged.connect(self.main.on_change_cmap)
        s_layout.addRow("Colormap", self.cmap_combo)
        self.time_cb = QCheckBox("Time scale")
        self.freq_cb = QCheckBox("Frequency scale")
        self.grid_cb = QCheckBox("Grid lines")
        s_layout.addRow(self.time_cb)
        s_layout.addRow(self.freq_cb)
        s_layout.addRow(self.grid_cb)
        layout.addWidget(spec_box)

        # ----- Stats -----
        stats_box = QGroupBox("Detection Info")
        stats_layout = QVBoxLayout(stats_box)
        self.stats_label = QLabel("Tracks detected: 0")
        stats_layout.addWidget(self.stats_label)
        layout.addWidget(stats_box)

    def _add_slider_spin(self, form, label, minv, maxv):
        slider = QSlider(Qt.Horizontal)
        spin = QSpinBox()
        slider.setRange(minv, maxv)
        spin.setRange(minv, maxv)
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        row = QHBoxLayout()
        row.addWidget(slider)
        row.addWidget(spin)
        form.addRow(label, row)
        return slider, spin

    def toggle(self, show: bool):
        self._expanded = show
        start = self.maximumHeight()
        end = 300 if show else 0
        self.animation.stop()
        self.animation.setStartValue(start)
        self.animation.setEndValue(end)
        self.animation.start()

    def update_stats(self, tracks: int, method: str, duration: float):
        self.stats_label.setText(
            f"Tracks detected: {tracks} | Method: {method} | {duration:.1f}s")
