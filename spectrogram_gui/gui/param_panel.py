from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout, QLabel, QGroupBox,
    QFormLayout, QComboBox, QSlider
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
        self.tracks_label.setObjectName("summaryLine")
        self.method_label = QLabel("Method: -")
        self.method_label.setObjectName("summaryLine")
        self.time_label = QLabel("Runtime: -")
        self.time_label.setObjectName("summaryLine")
        for lbl in (self.tracks_label, self.method_label, self.time_label):
            s_layout.addWidget(lbl)
        layout.addWidget(summary_box)

        # ----- Spectrogram Settings -----
        spec_box = QGroupBox("Spectrogram Settings")
        spec_box.setObjectName("stftSettings")
        s2_layout = QFormLayout(spec_box)

        self.nperseg_slider = QSlider(Qt.Horizontal)
        self.nperseg_slider.setRange(1, 64)  # 1=>256
        self.nperseg_value = QLabel()
        self.nperseg_value.setObjectName("sliderLabel")
        nperseg_row = QHBoxLayout()
        nperseg_row.addWidget(self.nperseg_slider, 1)
        nperseg_row.addWidget(self.nperseg_value)
        s2_layout.addRow("Window Size", nperseg_row)

        self.overlap_slider = QSlider(Qt.Horizontal)
        self.overlap_slider.setRange(0, 95)
        self.overlap_value = QLabel()
        self.overlap_value.setObjectName("sliderLabel")
        overlap_row = QHBoxLayout()
        overlap_row.addWidget(self.overlap_slider, 1)
        overlap_row.addWidget(self.overlap_value)
        s2_layout.addRow("Overlap", overlap_row)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["gray", "viridis", "magma", "inferno", "plasma"])
        s2_layout.addRow("Colormap", self.cmap_combo)

        right_layout = QVBoxLayout()
        right_layout.addWidget(spec_box)
        layout.addLayout(right_layout, 1)

    def bind_settings(self):
        p = self.main.spectrogram_params
        nperseg_val = p.get("window_size", 4096)
        self.nperseg_slider.setValue(int(nperseg_val / 256))
        self.nperseg_value.setText(str(nperseg_val))

        overlap_val = p.get("overlap", 75)
        self.overlap_slider.setValue(overlap_val)
        self.overlap_value.setText(f"{overlap_val} %")
        self.cmap_combo.setCurrentText(p.get("colormap", "magma"))

        def set_nperseg(v):
            value = max(1, v) * 256
            self.nperseg_value.setText(str(value))
            p["window_size"] = value

        def set_overlap(v):
            self.overlap_value.setText(f"{v} %")
            p["overlap"] = v

        self.nperseg_slider.valueChanged.connect(set_nperseg)
        self.overlap_slider.valueChanged.connect(set_overlap)
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
