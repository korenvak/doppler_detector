from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout, QLabel, QGroupBox,
    QSpinBox, QFormLayout, QComboBox
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
        summary_box.setObjectName("card")
        s_layout = QVBoxLayout(summary_box)
        self.tracks_label = QLabel("0 Tracks detected")
        self.tracks_label.setStyleSheet(
            "font-size:22px;font-weight:600;color:#7DD3FC")
        self.method_label = QLabel("Method: -")
        self.method_label.setStyleSheet("color:#CBD5E1")
        self.time_label = QLabel("Runtime: -")
        self.time_label.setStyleSheet("color:#CBD5E1")
        for lbl in (self.tracks_label, self.method_label, self.time_label):
            s_layout.addWidget(lbl)
        layout.addWidget(summary_box)

        # ----- Spectrogram Settings -----
        spec_box = QGroupBox("Spectrogram Settings")
        spec_box.setObjectName("card")
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
        right_layout.addWidget(spec_box)
        layout.addLayout(right_layout, 1)

    def bind_settings(self):
        p = self.main.spectrogram_params
        self.nperseg_spin.setValue(p.get("window_size", 4096))
        self.overlap_spin.setValue(p.get("overlap", 75))
        self.cmap_combo.setCurrentText(p.get("colormap", "magma"))

        self.nperseg_spin.valueChanged.connect(lambda v: p.__setitem__("window_size", v))
        self.overlap_spin.valueChanged.connect(lambda v: p.__setitem__("overlap", v))
        self.cmap_combo.currentTextChanged.connect(lambda t: p.__setitem__("colormap", t))

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
