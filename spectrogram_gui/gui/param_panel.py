from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout, QLabel, QGroupBox,
    QSlider, QSpinBox, QFormLayout, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt, QPropertyAnimation


class ParamPanel(QFrame):
    """Collapsible bottom drawer with filter and spectrogram settings."""

    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.setObjectName("paramPanel")
        self._expanded = False
        self.setMaximumHeight(0)
        self.animation = QPropertyAnimation(self, b"maximumHeight")
        self.animation.setDuration(250)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(20)

        # ----- Detection Summary -----
        summary_box = QGroupBox("Detection Summary")
        s_layout = QVBoxLayout(summary_box)
        self.tracks_label = QLabel("0 Tracks detected")
        self.tracks_label.setStyleSheet("font-size:14px;font-weight:600")
        self.method_label = QLabel("Method: -")
        self.time_label = QLabel("Runtime: -")
        for lbl in (self.tracks_label, self.method_label, self.time_label):
            s_layout.addWidget(lbl)
        layout.addWidget(summary_box)

        # ----- Filters -----
        filter_box = QGroupBox("Filters")
        f_layout = QFormLayout(filter_box)

        self.nlms_chk = QCheckBox("NLMS")
        self.nlms_mu, self.nlms_spin = self._add_slider_spin(f_layout, "NLMS \u03bc", 0, 100)
        f_layout.addRow(self.nlms_chk)
        
        self.lms_chk = QCheckBox("LMS")
        self.lms_mu, self.lms_spin = self._add_slider_spin(f_layout, "LMS \u03bc", 0, 100)
        f_layout.addRow(self.lms_chk)

        self.ale_chk = QCheckBox("ALE")
        self.ale_mu, self.ale_spin = self._add_slider_spin(f_layout, "ALE \u03bc", 0, 100)
        self.ale_delay_spin = QSpinBox()
        self.ale_delay_spin.setRange(0, 100)
        f_layout.addRow("ALE Delay", self.ale_delay_spin)
        f_layout.addRow(self.ale_chk)

        self.rls_chk = QCheckBox("RLS")
        self.rls_lambda, self.rls_spin = self._add_slider_spin(f_layout, "RLS \u03bb", 90, 100)
        f_layout.addRow(self.rls_chk)

        self.wiener_chk = QCheckBox("Wiener")
        self.wiener_noise, self.wiener_spin = self._add_slider_spin(f_layout, "Noise dB", -60, 0)
        f_layout.addRow(self.wiener_chk)

        # ----- Spectrogram Settings -----
        spec_box = QGroupBox("Spectrogram Settings")
        s2_layout = QFormLayout(spec_box)

        self.nperseg_spin = QSpinBox()
        self.nperseg_spin.setRange(256, 16384)
        self.nperseg_spin.setSingleStep(256)
        s2_layout.addRow("Window Size", self.nperseg_spin)

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 95)
        self.overlap_spin.setSuffix(" %")
        s2_layout.addRow("Overlap", self.overlap_spin)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["gray", "viridis", "magma", "inferno", "plasma"])
        s2_layout.addRow("Colormap", self.cmap_combo)

        right_layout = QVBoxLayout()
        right_layout.addWidget(filter_box)
        right_layout.addWidget(spec_box)
        layout.addLayout(right_layout, 1)

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

    def bind_settings(self):
        p = self.main.spectrogram_params
        self.nperseg_spin.setValue(p.get("window_size", 4096))
        self.overlap_spin.setValue(p.get("overlap", 75))
        self.cmap_combo.setCurrentText(p.get("colormap", "magma"))

        f = self.main.filter_params
        self.nlms_chk.setChecked(f.get("nlms_enabled", False))
        self.nlms_mu.setValue(f.get("nlms_mu", 10))
        self.lms_chk.setChecked(f.get("lms_enabled", False))
        self.lms_mu.setValue(f.get("lms_mu", 10))
        self.ale_chk.setChecked(f.get("ale_enabled", False))
        self.ale_mu.setValue(f.get("ale_mu", 10))
        self.ale_delay_spin.setValue(f.get("ale_delay", 0))
        self.rls_chk.setChecked(f.get("rls_enabled", False))
        self.rls_lambda.setValue(f.get("rls_lambda", 99))
        self.wiener_chk.setChecked(f.get("wiener_enabled", False))
        self.wiener_noise.setValue(f.get("wiener_noise", -20))

        self.nperseg_spin.valueChanged.connect(lambda v: p.__setitem__("window_size", v))
        self.overlap_spin.valueChanged.connect(lambda v: p.__setitem__("overlap", v))
        self.cmap_combo.currentTextChanged.connect(lambda t: p.__setitem__("colormap", t))

        self.nlms_chk.toggled.connect(lambda s: f.__setitem__("nlms_enabled", s))
        self.nlms_mu.valueChanged.connect(lambda v: f.__setitem__("nlms_mu", v))
        self.lms_chk.toggled.connect(lambda s: f.__setitem__("lms_enabled", s))
        self.lms_mu.valueChanged.connect(lambda v: f.__setitem__("lms_mu", v))
        self.ale_chk.toggled.connect(lambda s: f.__setitem__("ale_enabled", s))
        self.ale_mu.valueChanged.connect(lambda v: f.__setitem__("ale_mu", v))
        self.ale_delay_spin.valueChanged.connect(lambda v: f.__setitem__("ale_delay", v))
        self.rls_chk.toggled.connect(lambda s: f.__setitem__("rls_enabled", s))
        self.rls_lambda.valueChanged.connect(lambda v: f.__setitem__("rls_lambda", v))
        self.wiener_chk.toggled.connect(lambda s: f.__setitem__("wiener_enabled", s))
        self.wiener_noise.valueChanged.connect(lambda v: f.__setitem__("wiener_noise", v))

    def toggle(self, show: bool):
        self._expanded = show
        start = self.maximumHeight()
        end = 300 if show else 0
        self.animation.stop()
        self.animation.setStartValue(start)
        self.animation.setEndValue(end)
        self.animation.start()

    def update_stats(self, tracks: int, method: str, duration: float):
        self.tracks_label.setText(f"{tracks} Tracks detected")
        self.method_label.setText(f"Method: {method}")
        self.time_label.setText(f"Runtime: {duration:.1f}s")
