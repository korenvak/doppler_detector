from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel, QPushButton
import qtawesome as qta

class ParamPanel(QFrame):
    """Collapsible side panel for quick access to detection parameters."""
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.setObjectName("paramPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("Detection Settings")
        layout.addWidget(title)

        btn = QPushButton("Open Parameters")
        btn.setIcon(qta.icon('fa5s.sliders-h'))
        btn.clicked.connect(self.main.open_detector_params)
        layout.addWidget(btn)

        layout.addStretch()
