from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QLabel, QPushButton,
    QGroupBox, QHBoxLayout, QSlider
)
from PyQt5.QtCore import Qt, QPropertyAnimation
import qtawesome as qta

class ParamPanel(QFrame):
    """Collapsible side panel for quick access to detection parameters."""
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.setObjectName("paramPanel")
        self._expanded = False
        self.setMaximumHeight(0)
        self.animation = QPropertyAnimation(self, b"maximumHeight")
        self.animation.setDuration(200)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("Detection Settings")
        layout.addWidget(title)

        # Filter Settings
        filter_box = QGroupBox("Filter Settings")
        filter_layout = QHBoxLayout(filter_box)
        self.filter_slider = QSlider(Qt.Horizontal)
        filter_layout.addWidget(self.filter_slider)
        layout.addWidget(filter_box)

        # Detection Parameters
        det_box = QGroupBox("Detection Parameters")
        det_layout = QHBoxLayout(det_box)
        self.threshold_slider = QSlider(Qt.Horizontal)
        det_layout.addWidget(self.threshold_slider)
        layout.addWidget(det_box)

        btn = QPushButton("Open Full Dialog")
        btn.setIcon(qta.icon('fa5s.sliders-h'))
        btn.clicked.connect(self.main.open_detector_params)
        layout.addWidget(btn)

    def toggle(self, show: bool):
        self._expanded = show
        start = self.maximumHeight()
        end = 200 if show else 0
        self.animation.stop()
        self.animation.setStartValue(start)
        self.animation.setEndValue(end)
        self.animation.start()
