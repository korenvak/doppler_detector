from PyQt5.QtWidgets import (
    QDialog, QFormLayout, QSpinBox, QComboBox,
    QDialogButtonBox
)


class SpectrogramSettingsDialog(QDialog):
    """Dialog to tweak spectrogram display parameters."""

    def __init__(self, parent=None, params=None):
        super().__init__(parent)
        self.setWindowTitle("Spectrogram Settings")
        layout = QFormLayout(self)
        p = params or {}

        self.nperseg_spin = QSpinBox()
        self.nperseg_spin.setRange(256, 16384)
        self.nperseg_spin.setSingleStep(256)
        self.nperseg_spin.setValue(p.get("window_size", 4096))
        layout.addRow("Window Length", self.nperseg_spin)

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 95)
        self.overlap_spin.setSuffix(" %")
        self.overlap_spin.setValue(p.get("overlap", 75))
        layout.addRow("Overlap", self.overlap_spin)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["gray", "viridis", "magma", "inferno", "plasma"])
        self.cmap_combo.setCurrentText(p.get("colormap", "magma"))
        layout.addRow("Colormap", self.cmap_combo)

        self.gain_combo = QComboBox()
        self.gain_combo.addItems(["x1", "x2", "x4"])
        self.gain_combo.setCurrentText(f"x{p.get('gain', 1)}")
        layout.addRow("Gain", self.gain_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_params(self):
        return {
            "window_size": self.nperseg_spin.value(),
            "overlap": self.overlap_spin.value(),
            "colormap": self.cmap_combo.currentText(),
            "gain": int(self.gain_combo.currentText().lstrip('x'))
        }
