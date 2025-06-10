from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QTableView, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel, QStandardItem
import pandas as pd

class AnnotationEditorDialog(QDialog):
    def __init__(self, df: pd.DataFrame, parent=None):
        """
        Dialog to view and delete annotation rows before final CSV save.
        df: pandas DataFrame of all annotations (passed by reference).
        """
        super().__init__(parent)
        self.setWindowTitle("Edit Annotations")
        self.resize(600, 400)
        self.df = df  # Reference to the original DataFrame

        # Create a model for the QTableView
        self.model = QStandardItemModel(self)
        headers = list(df.columns)
        self.model.setColumnCount(len(headers))
        self.model.setHorizontalHeaderLabels(headers)

        # Populate the model with rows from the DataFrame
        for row_idx in range(df.shape[0]):
            for col_idx, col_name in enumerate(headers):
                item = QStandardItem(str(df.iloc[row_idx][col_name]))
                item.setEditable(False)
                self.model.setItem(row_idx, col_idx, item)

        # Create the table view and set its model
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)

        # Buttons: Delete Selected and Save & Close
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self.on_delete)

        self.save_btn = QPushButton("Save & Close")
        self.save_btn.clicked.connect(self.accept)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.delete_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.save_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(btn_layout)

    def on_delete(self):
        """
        Delete the selected row from both the model and the DataFrame.
        """
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return
        idx = selected[0].row()
        # Remove the row from the model
        self.model.removeRow(idx)
        # Remove the row from the DataFrame
        self.df.drop(self.df.index[idx], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
