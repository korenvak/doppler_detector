from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QSpinBox, QComboBox, QDialogButtonBox, QLabel
)

class ParamsDialog(QDialog):
    def __init__(self, parent=None, current_params=None):
        super().__init__(parent)
        self.setWindowTitle("Spectrogram Settings")

        self.window_size = 1024
        self.overlap = 50
        self.colormap = "gray"

        if current_params:
            self.window_size = current_params.get("window_size", 1024)
            self.overlap = current_params.get("overlap", 50)
            self.colormap = current_params.get("colormap", "gray")

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.window_input = QSpinBox()
        self.window_input.setRange(128, 8192)
        self.window_input.setValue(self.window_size)
        form.addRow("Window Size:", self.window_input)

        self.overlap_input = QSpinBox()
        self.overlap_input.setRange(0, 95)
        self.overlap_input.setSuffix(" %")
        self.overlap_input.setValue(self.overlap)
        form.addRow("Overlap (%):", self.overlap_input)

        self.colormap_input = QComboBox()
        self.colormap_input.addItems(["gray", "viridis", "magma", "inferno", "plasma"])
        self.colormap_input.setCurrentText(self.colormap)
        form.addRow("Colormap:", self.colormap_input)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_params(self):
        return {
            "window_size": self.window_input.value(),
            "overlap": self.overlap_input.value(),
            "colormap": self.colormap_input.currentText()
        }
