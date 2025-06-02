from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QHBoxLayout, QDialogButtonBox
)
from PyQt5.QtCore import Qt

class AutoDetectDialog(QDialog):
    def __init__(self, parent=None, param_grid=None):
        """
        Dialog for editing the Auto-Detect (Voting) parameter grid.
        param_grid: existing list of dicts, each with keys:
            "POWER_THRESHOLD", "PEAK_PROMINENCE",
            "MAX_FREQ_JUMP_HZ", "MIN_TRACK_LENGTH_FRAMES",
            "MAX_TRACK_FREQ_STD_HZ",
            "GAP_PROMINENCE_FACTOR", "GAP_POWER_FACTOR".
        """
        super().__init__(parent)
        self.setWindowTitle("Auto-Detect (Voting) Settings")

        layout = QVBoxLayout(self)

        headers = [
            "POWER_THRESHOLD",
            "PEAK_PROMINENCE",
            "MAX_FREQ_JUMP_HZ",
            "MIN_TRACK_LENGTH_FRAMES",
            "MAX_TRACK_FREQ_STD_HZ",
            "GAP_PROMINENCE_FACTOR",
            "GAP_POWER_FACTOR"
        ]
        self.table = QTableWidget()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

        if param_grid:
            self.table.setRowCount(len(param_grid))
            for row, cfg in enumerate(param_grid):
                self.table.setItem(row, 0, QTableWidgetItem(str(cfg.get("POWER_THRESHOLD", ""))))
                self.table.setItem(row, 1, QTableWidgetItem(str(cfg.get("PEAK_PROMINENCE", ""))))
                self.table.setItem(row, 2, QTableWidgetItem(str(cfg.get("MAX_FREQ_JUMP_HZ", ""))))
                self.table.setItem(row, 3, QTableWidgetItem(str(cfg.get("MIN_TRACK_LENGTH_FRAMES", ""))))
                self.table.setItem(row, 4, QTableWidgetItem(str(cfg.get("MAX_TRACK_FREQ_STD_HZ", ""))))
                self.table.setItem(row, 5, QTableWidgetItem(str(cfg.get("GAP_PROMINENCE_FACTOR", ""))))
                self.table.setItem(row, 6, QTableWidgetItem(str(cfg.get("GAP_POWER_FACTOR", ""))))
        else:
            self.table.setRowCount(0)

        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.add_row_button = QPushButton("Add Row")
        self.remove_row_button = QPushButton("Remove Selected")
        btn_layout.addWidget(self.add_row_button)
        btn_layout.addWidget(self.remove_row_button)
        layout.addLayout(btn_layout)

        self.add_row_button.clicked.connect(self.add_row)
        self.remove_row_button.clicked.connect(self.remove_selected)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.param_grid = param_grid or []

    def add_row(self):
        """
        Append a new row with default values.
        """
        row = self.table.rowCount()
        self.table.insertRow(row)
        defaults = ["0.20", "0.060", "15", "13", "70", "0.8", "0.8"]
        for col, val in enumerate(defaults):
            self.table.setItem(row, col, QTableWidgetItem(val))

    def remove_selected(self):
        """
        Remove any selected rows.
        """
        selected_rows = set(idx.row() for idx in self.table.selectedIndexes())
        for row in sorted(selected_rows, reverse=True):
            self.table.removeRow(row)

    def get_param_grid(self):
        """
        Build a list of dicts from the table contents.
        """
        grid = []
        for row in range(self.table.rowCount()):
            try:
                cfg = {
                    "POWER_THRESHOLD": float(self.table.item(row, 0).text()),
                    "PEAK_PROMINENCE": float(self.table.item(row, 1).text()),
                    "MAX_FREQ_JUMP_HZ": float(self.table.item(row, 2).text()),
                    "MIN_TRACK_LENGTH_FRAMES": int(self.table.item(row, 3).text()),
                    "MAX_TRACK_FREQ_STD_HZ": float(self.table.item(row, 4).text()),
                    "GAP_PROMINENCE_FACTOR": float(self.table.item(row, 5).text()),
                    "GAP_POWER_FACTOR": float(self.table.item(row, 6).text())
                }
            except Exception:
                continue
            grid.append(cfg)
        return grid

    def accept(self):
        """
        Save current table values to self.param_grid.
        """
        self.param_grid = self.get_param_grid()
        super().accept()
