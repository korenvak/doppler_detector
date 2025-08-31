"""
Modern Event Annotator with enhanced UI/UX
- Inline editing
- Visual timeline
- Tag categories with colors
- Keyboard shortcuts
- Export options
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QDialog, QDialogButtonBox,
    QLabel, QLineEdit, QTextEdit, QComboBox, QFrame, QMenu,
    QAction, QFileDialog, QMessageBox, QAbstractItemView,
    QStyledItemDelegate, QStyleOptionViewItem, QToolButton,
    QColorDialog, QCheckBox, QSpinBox, QStyle
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QRect, QPoint, QSize
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QFont, QIcon, QPalette
import pyqtgraph as pg
import qtawesome as qta


class TagColorDelegate(QStyledItemDelegate):
    """Custom delegate for rendering colored tags"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tag_colors = {
            "Detection": "#6366F1",
            "Anomaly": "#EF4444",
            "Event": "#10B981",
            "Note": "#F59E0B",
            "Custom": "#A78BFA"
        }
        
    def paint(self, painter, option, index):
        if index.column() == 4:  # Type column
            tag_type = index.data()
            color = self.tag_colors.get(tag_type, "#6B7280")
            
            # Draw background
            if option.state & QStyle.State_Selected:
                painter.fillRect(option.rect, QColor("#6366F1"))
            else:
                painter.fillRect(option.rect, QColor(option.palette.color(QPalette.Base)))
                
            # Draw colored tag
            tag_rect = QRect(
                option.rect.x() + 4,
                option.rect.y() + 4,
                option.rect.width() - 8,
                option.rect.height() - 8
            )
            
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(color)))
            painter.drawRoundedRect(tag_rect, 4, 4)
            
            # Draw text
            painter.setPen(QPen(QColor("#FFFFFF")))
            painter.setFont(QFont("Inter", 10, QFont.Bold))
            painter.drawText(tag_rect, Qt.AlignCenter, tag_type)
        else:
            super().paint(painter, option, index)


class ModernAnnotationDialog(QDialog):
    """Modern dialog for adding/editing annotations"""
    
    def __init__(self, start_time=None, end_time=None, annotation=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Annotation" if annotation is None else "Edit Annotation")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setup_ui()
        
        # Populate if editing
        if annotation:
            self.populate_from_annotation(annotation)
        elif start_time and end_time:
            self.start_edit.setText(start_time)
            self.end_edit.setText(end_time)
            
    def setup_ui(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Time range section
        time_frame = QFrame()
        time_frame.setObjectName("modernCard")
        time_layout = QVBoxLayout(time_frame)
        
        time_label = QLabel("Time Range")
        time_label.setStyleSheet("font-size: 16px; font-weight: 600; color: #A78BFA;")
        time_layout.addWidget(time_label)
        
        time_inputs = QHBoxLayout()
        
        self.start_edit = QLineEdit()
        self.start_edit.setPlaceholderText("Start time (HH:MM:SS.mmm)")
        time_inputs.addWidget(QLabel("Start:"))
        time_inputs.addWidget(self.start_edit)
        
        self.end_edit = QLineEdit()
        self.end_edit.setPlaceholderText("End time (HH:MM:SS.mmm)")
        time_inputs.addWidget(QLabel("End:"))
        time_inputs.addWidget(self.end_edit)
        
        time_layout.addLayout(time_inputs)
        layout.addWidget(time_frame)
        
        # Annotation details section
        details_frame = QFrame()
        details_frame.setObjectName("modernCard")
        details_layout = QVBoxLayout(details_frame)
        
        details_label = QLabel("Annotation Details")
        details_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #A78BFA;")
        details_layout.addWidget(details_label)
        
        # Type selector with predefined options
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        
        self.type_combo = QComboBox()
        self.type_combo.setEditable(True)
        self.type_combo.addItems(["Detection", "Anomaly", "Event", "Note", "Custom"])
        self.type_combo.setCurrentText("Detection")
        type_layout.addWidget(self.type_combo)
        
        # Color picker for custom types
        self.color_btn = QPushButton()
        self.color_btn.setMaximumWidth(40)
        self.color_btn.setStyleSheet("background: #6366F1; border-radius: 4px;")
        self.color_btn.clicked.connect(self.pick_color)
        type_layout.addWidget(self.color_btn)
        
        details_layout.addLayout(type_layout)
        
        # Confidence/Priority
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        
        self.confidence_spin = QSpinBox()
        self.confidence_spin.setRange(0, 100)
        self.confidence_spin.setValue(75)
        self.confidence_spin.setSuffix("%")
        conf_layout.addWidget(self.confidence_spin)
        
        conf_layout.addStretch()
        
        conf_layout.addWidget(QLabel("Priority:"))
        self.priority_combo = QComboBox()
        self.priority_combo.addItems(["Low", "Medium", "High", "Critical"])
        self.priority_combo.setCurrentText("Medium")
        conf_layout.addWidget(self.priority_combo)
        
        details_layout.addLayout(conf_layout)
        
        # Description
        desc_label = QLabel("Description:")
        details_layout.addWidget(desc_label)
        
        self.description_edit = QTextEdit()
        self.description_edit.setPlaceholderText("Enter detailed description...")
        self.description_edit.setMaximumHeight(100)
        details_layout.addWidget(self.description_edit)
        
        # Tags
        tags_layout = QHBoxLayout()
        tags_layout.addWidget(QLabel("Tags:"))
        
        self.tags_edit = QLineEdit()
        self.tags_edit.setPlaceholderText("Comma-separated tags (e.g., doppler, shift, anomaly)")
        tags_layout.addWidget(self.tags_edit)
        
        details_layout.addLayout(tags_layout)
        
        layout.addWidget(details_frame)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.setObjectName("accentButton")
        self.save_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
        
    def pick_color(self):
        """Open color picker"""
        color = QColorDialog.getColor(QColor("#6366F1"), self, "Select Tag Color")
        if color.isValid():
            self.color_btn.setStyleSheet(f"background: {color.name()}; border-radius: 4px;")
            
    def populate_from_annotation(self, annotation):
        """Populate fields from existing annotation"""
        self.start_edit.setText(annotation.get("Start", ""))
        self.end_edit.setText(annotation.get("End", ""))
        self.type_combo.setCurrentText(annotation.get("Type", "Detection"))
        self.description_edit.setPlainText(annotation.get("Description", ""))
        self.tags_edit.setText(annotation.get("Tags", ""))
        self.confidence_spin.setValue(annotation.get("Confidence", 75))
        self.priority_combo.setCurrentText(annotation.get("Priority", "Medium"))
        
    def get_annotation(self):
        """Get annotation data from dialog"""
        return {
            "Start": self.start_edit.text(),
            "End": self.end_edit.text(),
            "Type": self.type_combo.currentText(),
            "Description": self.description_edit.toPlainText(),
            "Tags": self.tags_edit.text(),
            "Confidence": self.confidence_spin.value(),
            "Priority": self.priority_combo.currentText()
        }


class ModernEventAnnotator(QWidget):
    """Modern event annotator with enhanced features"""
    
    annotation_added = pyqtSignal(dict)
    annotation_edited = pyqtSignal(int, dict)
    annotation_deleted = pyqtSignal(int)
    
    def __init__(self, canvas=None, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.csv_path = None
        self.metadata = {}
        self.start_time_pending = None
        self.current_items = []
        self.graphics_history = []
        
        # Initialize DataFrame with extended columns
        self.df = pd.DataFrame(columns=[
            "Start", "End", "Site", "Pixel", "Type", 
            "Description", "Tags", "Confidence", "Priority", "Snapshot"
        ])
        
        self.setup_ui()
        self.setup_shortcuts()
        
    def setup_ui(self):
        """Setup the modern UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header with controls
        header = self.create_header()
        layout.addWidget(header)
        
        # Annotation table
        self.table = self.create_table()
        layout.addWidget(self.table)
        
        # Statistics bar
        self.stats_bar = self.create_stats_bar()
        layout.addWidget(self.stats_bar)
        
    def create_header(self):
        """Create header with controls"""
        header = QFrame()
        header.setObjectName("modernCard")
        header.setMaximumHeight(50)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(8, 4, 8, 4)
        
        # Title
        title = QLabel("Event Annotations")
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: #F9FAFB;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Filter
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All Types", "Detection", "Anomaly", "Event", "Note"])
        self.filter_combo.currentTextChanged.connect(self.filter_annotations)
        layout.addWidget(self.filter_combo)
        
        # Search
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search annotations...")
        self.search_edit.setMaximumWidth(200)
        self.search_edit.textChanged.connect(self.search_annotations)
        layout.addWidget(self.search_edit)
        
        # Tools
        self.add_btn = QToolButton()
        self.add_btn.setIcon(qta.icon('mdi.plus', color='#10B981'))
        self.add_btn.setToolTip("Add Annotation (A)")
        self.add_btn.clicked.connect(self.add_annotation)
        layout.addWidget(self.add_btn)
        
        self.edit_btn = QToolButton()
        self.edit_btn.setIcon(qta.icon('mdi.pencil', color='#F59E0B'))
        self.edit_btn.setToolTip("Edit Selected (E)")
        self.edit_btn.clicked.connect(self.edit_selected)
        layout.addWidget(self.edit_btn)
        
        self.delete_btn = QToolButton()
        self.delete_btn.setIcon(qta.icon('mdi.delete', color='#EF4444'))
        self.delete_btn.setToolTip("Delete Selected (Del)")
        self.delete_btn.clicked.connect(self.delete_selected)
        layout.addWidget(self.delete_btn)
        
        # Export menu
        export_btn = QToolButton()
        export_btn.setIcon(qta.icon('mdi.export', color='#F9FAFB'))
        export_btn.setToolTip("Export")
        
        export_menu = QMenu(export_btn)
        export_menu.addAction("Export as CSV", lambda: self.export_annotations("csv"))
        export_menu.addAction("Export as JSON", lambda: self.export_annotations("json"))
        export_menu.addAction("Export as Excel", lambda: self.export_annotations("xlsx"))
        export_btn.setMenu(export_menu)
        export_btn.setPopupMode(QToolButton.InstantPopup)
        layout.addWidget(export_btn)
        
        return header
        
    def create_table(self):
        """Create modern annotation table"""
        table = QTableWidget()
        table.setObjectName("modernTable")
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        # Set columns
        table.setColumnCount(10)
        table.setHorizontalHeaderLabels([
            "Start", "End", "Duration", "Type", "Description",
            "Tags", "Confidence", "Priority", "Site", "Pixel"
        ])
        
        # Configure header
        header = table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(4, QHeaderView.Stretch)  # Description column stretches
        
        # Set column widths
        table.setColumnWidth(0, 100)  # Start
        table.setColumnWidth(1, 100)  # End
        table.setColumnWidth(2, 80)   # Duration
        table.setColumnWidth(3, 100)  # Type
        table.setColumnWidth(5, 150)  # Tags
        table.setColumnWidth(6, 80)   # Confidence
        table.setColumnWidth(7, 80)   # Priority
        table.setColumnWidth(8, 60)   # Site
        table.setColumnWidth(9, 60)   # Pixel
        
        # Set custom delegate for colored tags
        table.setItemDelegateForColumn(3, TagColorDelegate(table))
        
        # Connect signals
        table.itemDoubleClicked.connect(self.on_item_double_clicked)
        table.itemSelectionChanged.connect(self.on_selection_changed)
        
        return table
        
    def create_stats_bar(self):
        """Create statistics bar"""
        stats_bar = QFrame()
        stats_bar.setObjectName("modernCard")
        stats_bar.setMaximumHeight(40)
        
        layout = QHBoxLayout(stats_bar)
        layout.setContentsMargins(8, 4, 8, 4)
        
        self.stats_label = QLabel("0 annotations")
        self.stats_label.setStyleSheet("color: #9CA3AF; font-size: 12px;")
        layout.addWidget(self.stats_label)
        
        layout.addStretch()
        
        # Type distribution
        self.type_stats = QLabel("")
        self.type_stats.setStyleSheet("color: #9CA3AF; font-size: 12px;")
        layout.addWidget(self.type_stats)
        
        return stats_bar
        
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        
        # Add annotation: A
        QShortcut(QKeySequence("A"), self, self.add_annotation)
        
        # Edit selected: E
        QShortcut(QKeySequence("E"), self, self.edit_selected)
        
        # Delete selected: Del
        QShortcut(QKeySequence(Qt.Key_Delete), self, self.delete_selected)
        
        # Copy: Ctrl+C
        QShortcut(QKeySequence.Copy, self, self.copy_selected)
        
        # Paste: Ctrl+V
        QShortcut(QKeySequence.Paste, self, self.paste_annotation)
        
    def set_metadata(self, site, pixel, file_start):
        """Set metadata for annotations"""
        self.metadata = {
            "site": site,
            "pixel": pixel,
            "file_start": file_start
        }
        
    def set_csv_path(self, path):
        """Set CSV path and load existing annotations"""
        self.csv_path = path
        if os.path.exists(path):
            try:
                self.df = pd.read_csv(path)
                self.refresh_table()
                self.update_stats()
            except Exception as e:
                print(f"Error loading CSV: {e}")
                
    def on_click(self, rel_sec, event):
        """Handle canvas clicks for annotation"""
        if not self.metadata.get("file_start") or not self.canvas:
            return
            
        file_start = self.metadata["file_start"]
        
        # First click - set start
        if self.start_time_pending is None:
            self.start_time_pending = rel_sec
            
            # Draw start marker
            if self.canvas and self.canvas.times is not None:
                x_pos = self.canvas.times[0] + rel_sec
                
                line = pg.InfiniteLine(
                    pos=x_pos,
                    angle=90,
                    pen=pg.mkPen('#10B981', width=2, style=Qt.DashLine)
                )
                
                label = pg.TextItem(
                    "Start",
                    anchor=(0.5, 1),
                    color='#10B981'
                )
                label.setPos(x_pos, self.canvas.freqs[-1] if self.canvas.freqs is not None else 0)
                
                self.canvas.plot.addItem(line)
                self.canvas.plot.addItem(label)
                self.current_items = [line, label]
                
        # Second click - set end and create annotation
        else:
            start_rel = self.start_time_pending
            end_rel = rel_sec
            self.start_time_pending = None
            
            # Ensure start < end
            if end_rel < start_rel:
                start_rel, end_rel = end_rel, start_rel
                
            # Calculate times
            start_dt = file_start + timedelta(seconds=start_rel)
            end_dt = file_start + timedelta(seconds=end_rel)
            
            # Format times
            start_str = start_dt.strftime("%H:%M:%S.%f")[:-3]
            end_str = end_dt.strftime("%H:%M:%S.%f")[:-3]
            
            # Open annotation dialog
            dialog = ModernAnnotationDialog(start_str, end_str, parent=self)
            if dialog.exec_() == QDialog.Accepted:
                annotation = dialog.get_annotation()
                
                # Add metadata
                annotation["Site"] = self.metadata.get("site", "")
                annotation["Pixel"] = self.metadata.get("pixel", "")
                annotation["Snapshot"] = ""  # Will be filled by snapshot system
                
                # Add to DataFrame
                self.df = pd.concat([self.df, pd.DataFrame([annotation])], ignore_index=True)
                
                # Save and refresh
                self.save_annotations()
                self.refresh_table()
                self.update_stats()
                
                # Draw end marker
                if self.canvas and self.canvas.times is not None:
                    x_pos = self.canvas.times[0] + end_rel
                    
                    line = pg.InfiniteLine(
                        pos=x_pos,
                        angle=90,
                        pen=pg.mkPen('#10B981', width=2, style=Qt.DashLine)
                    )
                    
                    label = pg.TextItem(
                        "End",
                        anchor=(0.5, 1),
                        color='#10B981'
                    )
                    label.setPos(x_pos, self.canvas.freqs[-1] if self.canvas.freqs is not None else 0)
                    
                    self.canvas.plot.addItem(line)
                    self.canvas.plot.addItem(label)
                    
                    # Store graphics items
                    self.graphics_history.append(self.current_items + [line, label])
                    
                # Emit signal
                self.annotation_added.emit(annotation)
                
            # Clear temporary items
            for item in self.current_items:
                if self.canvas:
                    self.canvas.plot.removeItem(item)
            self.current_items = []
            
    def add_annotation(self):
        """Add new annotation manually"""
        dialog = ModernAnnotationDialog(parent=self)
        if dialog.exec_() == QDialog.Accepted:
            annotation = dialog.get_annotation()
            
            # Add metadata
            annotation["Site"] = self.metadata.get("site", "")
            annotation["Pixel"] = self.metadata.get("pixel", "")
            annotation["Snapshot"] = ""
            
            # Add to DataFrame
            self.df = pd.concat([self.df, pd.DataFrame([annotation])], ignore_index=True)
            
            # Save and refresh
            self.save_annotations()
            self.refresh_table()
            self.update_stats()
            
            # Emit signal
            self.annotation_added.emit(annotation)
            
    def edit_selected(self):
        """Edit selected annotation"""
        row = self.table.currentRow()
        if row >= 0:
            annotation = self.df.iloc[row].to_dict()
            dialog = ModernAnnotationDialog(annotation=annotation, parent=self)
            
            if dialog.exec_() == QDialog.Accepted:
                updated = dialog.get_annotation()
                
                # Update DataFrame
                for key, value in updated.items():
                    if key in self.df.columns:
                        self.df.at[row, key] = value
                        
                # Save and refresh
                self.save_annotations()
                self.refresh_table()
                self.update_stats()
                
                # Emit signal
                self.annotation_edited.emit(row, updated)
                
    def delete_selected(self):
        """Delete selected annotations"""
        rows = set()
        for item in self.table.selectedItems():
            rows.add(item.row())
            
        if rows:
            reply = QMessageBox.question(
                self,
                "Delete Annotations",
                f"Delete {len(rows)} selected annotation(s)?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Remove rows from DataFrame
                self.df = self.df.drop(list(rows)).reset_index(drop=True)
                
                # Save and refresh
                self.save_annotations()
                self.refresh_table()
                self.update_stats()
                
                # Emit signals
                for row in rows:
                    self.annotation_deleted.emit(row)
                    
    def on_item_double_clicked(self, item):
        """Handle double-click to edit"""
        self.edit_selected()
        
    def on_selection_changed(self):
        """Handle selection change"""
        has_selection = len(self.table.selectedItems()) > 0
        self.edit_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)
        
    def filter_annotations(self, filter_type):
        """Filter annotations by type"""
        for row in range(self.table.rowCount()):
            if filter_type == "All Types":
                self.table.setRowHidden(row, False)
            else:
                type_item = self.table.item(row, 3)
                if type_item:
                    self.table.setRowHidden(row, type_item.text() != filter_type)
                    
    def search_annotations(self, search_text):
        """Search annotations"""
        search_text = search_text.lower()
        
        for row in range(self.table.rowCount()):
            # Search in description and tags
            desc_item = self.table.item(row, 4)
            tags_item = self.table.item(row, 5)
            
            match = False
            if desc_item and search_text in desc_item.text().lower():
                match = True
            if tags_item and search_text in tags_item.text().lower():
                match = True
                
            self.table.setRowHidden(row, not match if search_text else False)
            
    def refresh_table(self):
        """Refresh table with DataFrame content"""
        self.table.setRowCount(len(self.df))
        
        for i, row in self.df.iterrows():
            # Calculate duration if possible
            try:
                start = pd.to_datetime(row.get("Start", ""))
                end = pd.to_datetime(row.get("End", ""))
                duration = (end - start).total_seconds()
                duration_str = f"{duration:.2f}s"
            except:
                duration_str = ""
                
            # Set items
            self.table.setItem(i, 0, QTableWidgetItem(str(row.get("Start", ""))))
            self.table.setItem(i, 1, QTableWidgetItem(str(row.get("End", ""))))
            self.table.setItem(i, 2, QTableWidgetItem(duration_str))
            self.table.setItem(i, 3, QTableWidgetItem(str(row.get("Type", ""))))
            self.table.setItem(i, 4, QTableWidgetItem(str(row.get("Description", ""))))
            self.table.setItem(i, 5, QTableWidgetItem(str(row.get("Tags", ""))))
            self.table.setItem(i, 6, QTableWidgetItem(f"{row.get('Confidence', 0)}%"))
            self.table.setItem(i, 7, QTableWidgetItem(str(row.get("Priority", ""))))
            self.table.setItem(i, 8, QTableWidgetItem(str(row.get("Site", ""))))
            self.table.setItem(i, 9, QTableWidgetItem(str(row.get("Pixel", ""))))
            
    def update_stats(self):
        """Update statistics"""
        total = len(self.df)
        self.stats_label.setText(f"{total} annotation{'s' if total != 1 else ''}")
        
        # Type distribution
        if total > 0:
            type_counts = self.df["Type"].value_counts()
            type_str = " | ".join([f"{t}: {c}" for t, c in type_counts.items()[:3]])
            self.type_stats.setText(type_str)
        else:
            self.type_stats.setText("")
            
    def save_annotations(self):
        """Save annotations to CSV"""
        if self.csv_path:
            try:
                os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
                self.df.to_csv(self.csv_path, index=False)
            except Exception as e:
                print(f"Error saving annotations: {e}")
                
    def export_annotations(self, format):
        """Export annotations in different formats"""
        if len(self.df) == 0:
            QMessageBox.information(self, "No Data", "No annotations to export.")
            return
            
        # Get save path
        if format == "csv":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export CSV", "", "CSV Files (*.csv)"
            )
            if path:
                self.df.to_csv(path, index=False)
                
        elif format == "json":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export JSON", "", "JSON Files (*.json)"
            )
            if path:
                self.df.to_json(path, orient="records", indent=2)
                
        elif format == "xlsx":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Excel", "", "Excel Files (*.xlsx)"
            )
            if path:
                self.df.to_excel(path, index=False)
                
        if path:
            QMessageBox.information(
                self, "Export Complete", 
                f"Annotations exported to:\n{path}"
            )
            
    def copy_selected(self):
        """Copy selected annotations to clipboard"""
        rows = set()
        for item in self.table.selectedItems():
            rows.add(item.row())
            
        if rows:
            selected_df = self.df.iloc[list(rows)]
            selected_df.to_clipboard(index=False)
            
    def paste_annotation(self):
        """Paste annotation from clipboard"""
        try:
            clipboard_df = pd.read_clipboard()
            if not clipboard_df.empty:
                self.df = pd.concat([self.df, clipboard_df], ignore_index=True)
                self.save_annotations()
                self.refresh_table()
                self.update_stats()
        except:
            pass
            
    def undo_last(self):
        """Undo last annotation"""
        if not self.df.empty:
            self.df = self.df.iloc[:-1].copy()
            self.save_annotations()
            self.refresh_table()
            self.update_stats()
            
            # Remove graphics
            if self.graphics_history:
                items = self.graphics_history.pop()
                for item in items:
                    if self.canvas:
                        try:
                            self.canvas.plot.removeItem(item)
                        except:
                            pass
                            
    def count(self):
        """Get annotation count"""
        return len(self.df)