from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QVBoxLayout, QLabel, QGroupBox,
    QSlider, QSpinBox, QFormLayout
)
from PyQt5.QtCore import Qt, QPropertyAnimation


class ParamPanel(QFrame):
    """Collapsible bottom drawer showing detection summary and parameters."""

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

        # ----- Summary -----
        summary_box = QGroupBox("Detection Summary")
        s_layout = QVBoxLayout(summary_box)
        self.tracks_label = QLabel("0 Tracks detected")
        self.method_label = QLabel("Method: -")
        self.time_label = QLabel("Runtime: -")
        for lbl in (self.tracks_label, self.method_label, self.time_label):
            s_layout.addWidget(lbl)
        layout.addWidget(summary_box)

        # ----- Parameters -----
        det_box = QGroupBox("Detection Parameters")
        d_layout = QFormLayout(det_box)
        self.threshold_slider, self.threshold_spin = self._add_slider_spin(d_layout, "Power Threshold", 0, 100)
        self.prom_slider, self.prom_spin = self._add_slider_spin(d_layout, "Peak Prominence", 0, 50)
        self.jump_slider, self.jump_spin = self._add_slider_spin(d_layout, "Max Hz Jump", 0, 400)
        self.len_slider, self.len_spin = self._add_slider_spin(d_layout, "Track Min Length", 0, 10)
        self.std_slider, self.std_spin = self._add_slider_spin(d_layout, "Track Freq Std Max", 0, 50)
        layout.addWidget(det_box, 1)

        self.detector = None

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

    def bind_detector(self, detector):
        self.detector = detector
        self.threshold_slider.setValue(int(detector.power_threshold * 100))
        self.prom_slider.setValue(int(detector.peak_prominence * 100))
        self.jump_slider.setValue(int(detector.max_freq_jump_hz))
        self.len_slider.setValue(detector.min_track_length_frames)
        self.std_slider.setValue(int(detector.max_track_freq_std_hz))

        self.threshold_slider.valueChanged.connect(lambda v: setattr(detector, 'power_threshold', v / 100))
        self.prom_slider.valueChanged.connect(lambda v: setattr(detector, 'peak_prominence', v / 100))
        self.jump_slider.valueChanged.connect(lambda v: setattr(detector, 'max_freq_jump_hz', v))
        self.len_slider.valueChanged.connect(lambda v: setattr(detector, 'min_track_length_frames', v))
        self.std_slider.valueChanged.connect(lambda v: setattr(detector, 'max_track_freq_std_hz', v))

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
