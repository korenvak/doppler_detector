import multiprocessing
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QSpinBox, QCheckBox, QHBoxLayout
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

class ControlsPanel(QWidget):
    def __init__(self, parent=None):
        """
        ControlsPanel now contains:
          1. Open FLAC File button
          2. Detection Parameters button
          3. Auto-Detect Settings button
          4. Enable Multiprocessing checkbox
          5. Num Cores SpinBox
          6. Use Auto-Voting checkbox
          7. Run Detection button
          8. Export CSV button
          9. Classify Event (Num of Harmonics) SpinBox
         10. Undo Edit button
         11. Add Manual Event button
        """
        super().__init__(parent)

        layout = QVBoxLayout()

        # 1) Open FLAC File button
        self.open_button = QPushButton("Open FLAC File")
        self.open_button.setToolTip("Browse and select a FLAC or WAV audio file")
        layout.addWidget(self.open_button)

        # 2) Detection Parameters button
        self.params_button = QPushButton("Detection Parameters")
        self.params_button.setToolTip("Open dialog to set detection parameters")
        layout.addWidget(self.params_button)

        # 3) Auto-Detect Settings button
        self.auto_button = QPushButton("Auto-Detect Settings")
        self.auto_button.setToolTip("Open dialog to set auto-detect (voting) parameters")
        layout.addWidget(self.auto_button)

        # 4) Enable Multiprocessing checkbox
        self.mp_checkbox = QCheckBox("Enable Multiprocessing")
        self.mp_checkbox.setToolTip("Enable multiprocessing in Auto-Detect")
        layout.addWidget(self.mp_checkbox)

        # 5) Num Cores SpinBox
        layout.addWidget(QLabel("Num Cores (for Auto-Detect):"))
        self.num_cores_spin = QSpinBox()
        self.num_cores_spin.setRange(1, multiprocessing.cpu_count())
        self.num_cores_spin.setValue(1)
        self.num_cores_spin.setToolTip("Number of CPU cores to use in Auto-Detect")
        layout.addWidget(self.num_cores_spin)

        # 6) Use Auto-Voting checkbox
        self.voting_checkbox = QCheckBox("Use Auto-Voting (multi-config)")
        self.voting_checkbox.setToolTip("Run detection over a grid of parameters and merge results")
        layout.addWidget(self.voting_checkbox)

        # 7 + 8) Run Detection & Export CSV buttons
        run_export_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Detection")
        self.run_button.setToolTip("Start Doppler detection with current parameters")
        run_export_layout.addWidget(self.run_button)

        self.save_csv_button = QPushButton("Export CSV")
        self.save_csv_button.setToolTip("Export detected events to a CSV file")
        run_export_layout.addWidget(self.save_csv_button)
        layout.addLayout(run_export_layout)

        # 9) Classify Event (Num of Harmonics) SpinBox
        layout.addWidget(QLabel("Classify Event (Num of Harmonics):"))
        self.harmonics_spin = QSpinBox()
        self.harmonics_spin.setMinimum(0)
        self.harmonics_spin.setMaximum(10)
        self.harmonics_spin.setValue(1)
        self.harmonics_spin.setToolTip("Manual number of harmonics to assign to each event")
        layout.addWidget(self.harmonics_spin)

        # 10) Undo button
        self.undo_button = QPushButton("Undo")
        self.undo_button.setToolTip("Undo last edit (delete or manual-add)")
        layout.addWidget(self.undo_button)

        # 11) Add Manual Event button
        self.add_button = QPushButton("Add Event")
        self.add_button.setToolTip("Enter 'Add Mode' to manually annotate a new track")
        layout.addWidget(self.add_button)

        layout.addStretch()
        self.setLayout(layout)
