"""
Enhanced Spectrogram Canvas with Waveform Display and Event Marking
Includes expandable/collapsible waveform panel and event detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QToolBar, 
    QInputDialog, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt, Signal, QTimer, Slot
from PySide6.QtGui import QAction, QKeySequence
import qtawesome as qta

from .spectrogram_canvas import OptimizedSpectrogramCanvas, OptimizedViewBox, ModernTimeAxis


class WaveformDisplay(QWidget):
    """
    Waveform display that syncs with spectrogram
    """
    click_callback = Signal(float, object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
        # Data storage
        self.audio_data = None
        self.sample_rate = None
        self.times = None
        
        # Playback marker
        self.playback_line = None
        
    def setup_ui(self):
        """Setup the waveform display"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create graphics view
        self.graphics_view = pg.GraphicsLayoutWidget()
        self.graphics_view.setBackground((15, 15, 30, 200))
        layout.addWidget(self.graphics_view)
        
        # Create time axis (shared with spectrogram)
        self.time_axis = ModernTimeAxis(orientation="bottom")
        
        # Create custom viewbox for synchronized zoom
        self.viewbox = OptimizedViewBox()
        
        # Create plot
        self.plot = self.graphics_view.addPlot(
            viewBox=self.viewbox,
            axisItems={"bottom": self.time_axis}
        )
        
        # Configure appearance
        self.plot.showGrid(x=True, y=False, alpha=0.1)
        self.plot.setLabel('left', 'Amplitude', 
                          **{'color': '#9CA3AF', 'font-size': '10pt'})
        self.plot.setLabel('bottom', 'Time', units='s',
                          **{'color': '#9CA3AF', 'font-size': '10pt'})
        
        # Create waveform curve
        self.waveform_curve = pg.PlotCurveItem(
            pen=pg.mkPen('#8B5CF6', width=1)
        )
        self.plot.addItem(self.waveform_curve)
        
        # Create envelope curves for better visualization
        self.envelope_upper = pg.PlotCurveItem(
            pen=pg.mkPen('#6366F1', width=0.5, style=Qt.PenStyle.DashLine)
        )
        self.envelope_lower = pg.PlotCurveItem(
            pen=pg.mkPen('#6366F1', width=0.5, style=Qt.PenStyle.DashLine)
        )
        self.plot.addItem(self.envelope_upper)
        self.plot.addItem(self.envelope_lower)
        
        # Setup mouse tracking
        self.plot.scene().sigMouseClicked.connect(self.on_mouse_click)
        
    def set_audio_data(self, audio_data, sample_rate):
        """Set audio data for waveform display"""
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        
        if audio_data is None:
            return
            
        # Create time axis
        self.times = np.arange(len(audio_data)) / sample_rate
        
        # Downsample for display if needed
        max_points = 10000
        if len(audio_data) > max_points:
            # Intelligent downsampling - keep peaks
            downsample_factor = len(audio_data) // max_points
            
            # Reshape and get min/max for each chunk
            padding = downsample_factor - (len(audio_data) % downsample_factor)
            padded_audio = np.pad(audio_data, (0, padding), mode='constant')
            reshaped = padded_audio.reshape(-1, downsample_factor)
            
            # Get envelope
            max_vals = np.max(reshaped, axis=1)
            min_vals = np.min(reshaped, axis=1)
            
            # Downsample time
            display_times = self.times[::downsample_factor][:len(max_vals)]
            
            # Display envelope
            self.envelope_upper.setData(display_times, max_vals)
            self.envelope_lower.setData(display_times, min_vals)
            
            # Display average
            avg_vals = np.mean(reshaped, axis=1)
            self.waveform_curve.setData(display_times, avg_vals)
        else:
            # Display full waveform
            self.waveform_curve.setData(self.times, audio_data)
            self.envelope_upper.clear()
            self.envelope_lower.clear()
            
        # Set data bounds
        self.viewbox.set_data_bounds(
            0, self.times[-1],
            np.min(audio_data) * 1.1, np.max(audio_data) * 1.1
        )
        
        # Auto range
        self.viewbox.autoRange()
        
    def set_playback_position(self, position):
        """Update playback position marker"""
        if self.playback_line is None:
            self.playback_line = pg.InfiniteLine(
                angle=90, movable=False,
                pen=pg.mkPen('#EF4444', width=2)
            )
            self.plot.addItem(self.playback_line)
            
        self.playback_line.setPos(position)
        
    def on_mouse_click(self, event):
        """Handle mouse clicks"""
        pos = event.scenePos()
        mouse_point = self.viewbox.mapSceneToView(pos)
        x = mouse_point.x()
        
        if self.times is not None and 0 <= x <= self.times[-1]:
            self.click_callback.emit(x, event)
            
    def link_x_axis(self, other_viewbox):
        """Link X axis with another viewbox (for synchronized zoom)"""
        self.viewbox.setXLink(other_viewbox)


class EventAnnotator:
    """
    Event annotation system for marking and exporting events
    """
    def __init__(self, canvas):
        self.canvas = canvas
        self.csv_path = None
        self.df = pd.DataFrame(
            columns=["Start", "End", "Site", "Pixel", "Type", 
                    "Description", "Frequency", "Amplitude"]
        )
        self.start_time_pending = None
        self.metadata = {}
        self.graphics_items = []
        self.marking_enabled = False
        
    def set_metadata(self, site="", pixel="", file_start=None):
        """Set metadata for annotations"""
        self.metadata["site"] = site
        self.metadata["pixel"] = pixel
        self.metadata["file_start"] = file_start or datetime.now()
        
    def set_csv_path(self, path):
        """Set CSV output path and load existing annotations"""
        self.csv_path = path
        if os.path.exists(path):
            try:
                self.df = pd.read_csv(path)
                self.display_existing_annotations()
            except Exception as e:
                print(f"Error loading annotations: {e}")
                
    def display_existing_annotations(self):
        """Display previously saved annotations on the canvas"""
        # Clear old graphics
        for item in self.graphics_items:
            try:
                self.canvas.plot.removeItem(item)
            except:
                pass
        self.graphics_items.clear()
        
        # Draw each annotation
        for _, row in self.df.iterrows():
            try:
                # Parse times
                start_dt = pd.to_datetime(row['Start'])
                end_dt = pd.to_datetime(row['End'])
                
                # Convert to relative seconds if we have file_start
                if self.metadata.get("file_start"):
                    start_rel = (start_dt - self.metadata["file_start"]).total_seconds()
                    end_rel = (end_dt - self.metadata["file_start"]).total_seconds()
                    
                    # Draw region
                    region = pg.LinearRegionItem(
                        values=(start_rel, end_rel),
                        brush=pg.mkBrush(139, 92, 246, 30),
                        pen=pg.mkPen('#8B5CF6', width=1)
                    )
                    region.setMovable(False)
                    self.canvas.plot.addItem(region)
                    self.graphics_items.append(region)
                    
                    # Add label
                    label = pg.TextItem(
                        f"{row['Type']}: {row['Description'][:20]}",
                        color=(255, 255, 255, 200),
                        anchor=(0, 1)
                    )
                    label.setPos(start_rel, self.canvas.freqs[-1] if self.canvas.freqs is not None else 0)
                    self.canvas.plot.addItem(label)
                    self.graphics_items.append(label)
            except Exception as e:
                print(f"Error displaying annotation: {e}")
                
    def on_click(self, rel_sec, event):
        """Handle click events for marking"""
        if not self.marking_enabled:
            return
            
        file_start = self.metadata.get("file_start")
        if file_start is None:
            return
            
        # First click - start marking
        if self.start_time_pending is None:
            self.start_time_pending = rel_sec
            
            # Draw start line
            line = pg.InfiniteLine(
                pos=rel_sec,
                angle=90,
                pen=pg.mkPen(color=(0, 255, 255), width=2, style=Qt.PenStyle.DashLine)
            )
            self.canvas.plot.addItem(line)
            self.graphics_items.append(line)
            
            # Add label
            label = pg.TextItem(
                "Start",
                color=(0, 255, 255, 200),
                anchor=(0.5, 1)
            )
            y_pos = self.canvas.freqs[-1] * 0.95 if self.canvas.freqs is not None else 0
            label.setPos(rel_sec, y_pos)
            self.canvas.plot.addItem(label)
            self.graphics_items.append(label)
            
        # Second click - end marking
        else:
            start_rel = self.start_time_pending
            end_rel = rel_sec
            self.start_time_pending = None
            
            if end_rel < start_rel:
                start_rel, end_rel = end_rel, start_rel
                
            # Calculate absolute times
            start_dt = file_start + timedelta(seconds=start_rel)
            end_dt = file_start + timedelta(seconds=end_rel)
            
            # Get annotation details
            ev_type, ok1 = QInputDialog.getText(
                None, "Event Type", "Enter event type:"
            )
            if not ok1 or not ev_type.strip():
                return
                
            desc, ok2 = QInputDialog.getText(
                None, "Description", "Enter description:"
            )
            if not ok2:
                desc = ""
                
            # Get frequency and amplitude at cursor position
            freq = 0
            amp = 0
            if self.canvas.Sxx is not None and self.canvas.freqs is not None:
                # Find nearest time index
                time_idx = np.searchsorted(self.canvas.times, start_rel)
                if 0 <= time_idx < self.canvas.Sxx.shape[1]:
                    # Get peak frequency in this time slice
                    spectrum = self.canvas.Sxx[:, time_idx]
                    peak_idx = np.argmax(spectrum)
                    freq = self.canvas.freqs[peak_idx]
                    amp = 10 * np.log10(spectrum[peak_idx] + 1e-10)
                    
            # Create annotation row
            row = {
                "Start": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "End": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "Site": self.metadata.get("site", ""),
                "Pixel": self.metadata.get("pixel", ""),
                "Type": ev_type,
                "Description": desc,
                "Frequency": f"{freq:.1f}",
                "Amplitude": f"{amp:.1f}"
            }
            
            # Add to dataframe
            self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
            
            # Save CSV
            if self.csv_path:
                try:
                    os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
                    self.df.to_csv(self.csv_path, index=False)
                    print(f"Saved annotation to {self.csv_path}")
                except Exception as e:
                    print(f"Error saving CSV: {e}")
                    
            # Draw region
            region = pg.LinearRegionItem(
                values=(start_rel, end_rel),
                brush=pg.mkBrush(139, 92, 246, 50),
                pen=pg.mkPen('#8B5CF6', width=2)
            )
            region.setMovable(False)
            self.canvas.plot.addItem(region)
            self.graphics_items.append(region)
            
            # Add label
            label = pg.TextItem(
                f"{ev_type}: {desc[:20]}",
                color=(255, 255, 255, 200),
                anchor=(0, 1)
            )
            y_pos = self.canvas.freqs[-1] * 0.95 if self.canvas.freqs is not None else 0
            label.setPos(start_rel, y_pos)
            self.canvas.plot.addItem(label)
            self.graphics_items.append(label)
            
    def toggle_marking(self, enabled):
        """Toggle event marking mode"""
        self.marking_enabled = enabled
        self.start_time_pending = None
        
    def export_csv(self, path=None):
        """Export annotations to CSV"""
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                None, "Export Annotations", "", "CSV Files (*.csv)"
            )
        if path:
            try:
                self.df.to_csv(path, index=False)
                return True
            except Exception as e:
                QMessageBox.warning(None, "Export Error", str(e))
                return False
        return False
        
    def clear_annotations(self):
        """Clear all annotations"""
        self.df = pd.DataFrame(
            columns=["Start", "End", "Site", "Pixel", "Type", 
                    "Description", "Frequency", "Amplitude"]
        )
        for item in self.graphics_items:
            try:
                self.canvas.plot.removeItem(item)
            except:
                pass
        self.graphics_items.clear()
        
    def undo_last(self):
        """Undo last annotation"""
        if not self.df.empty:
            self.df = self.df.iloc[:-1].copy()
            if self.csv_path:
                try:
                    self.df.to_csv(self.csv_path, index=False)
                except:
                    pass
            # Remove last graphics items
            if len(self.graphics_items) >= 2:
                for _ in range(2):  # Remove region and label
                    if self.graphics_items:
                        item = self.graphics_items.pop()
                        try:
                            self.canvas.plot.removeItem(item)
                        except:
                            pass


class SpectrogramWithWaveform(QWidget):
    """
    Combined spectrogram and waveform display with event marking
    """
    position_clicked = Signal(float)
    event_marked = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
        # Event annotator
        self.event_annotator = EventAnnotator(self.spectrogram)
        
        # Detector reference
        self.detector = None
        
        # Connect signals
        self.setup_connections()
        
    def setup_ui(self):
        """Setup the UI with spectrogram and waveform"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create toolbar for controls
        self.toolbar = QToolBar()
        self.toolbar.setStyleSheet("""
            QToolBar {
                background: rgba(0, 0, 0, 0.2);
                border: none;
                padding: 4px;
            }
            QToolButton {
                background: transparent;
                border: none;
                padding: 4px;
                margin: 2px;
                border-radius: 4px;
            }
            QToolButton:hover {
                background: rgba(255, 255, 255, 0.1);
            }
            QToolButton:checked {
                background: rgba(139, 92, 246, 0.3);
            }
        """)
        
        # Add toolbar actions
        self.mark_action = QAction(qta.icon('fa5s.map-marker-alt', color='#9CA3AF'), "Mark Event", self)
        self.mark_action.setCheckable(True)
        self.mark_action.triggered.connect(self.toggle_marking)
        self.toolbar.addAction(self.mark_action)
        
        self.detect_action = QAction(qta.icon('fa5s.search', color='#9CA3AF'), "Run Detection", self)
        self.detect_action.triggered.connect(self.run_detection)
        self.toolbar.addAction(self.detect_action)
        
        self.export_action = QAction(qta.icon('fa5s.file-export', color='#9CA3AF'), "Export CSV", self)
        self.export_action.triggered.connect(self.export_annotations)
        self.toolbar.addAction(self.export_action)
        
        self.clear_action = QAction(qta.icon('fa5s.trash', color='#9CA3AF'), "Clear Marks", self)
        self.clear_action.triggered.connect(self.clear_annotations)
        self.toolbar.addAction(self.clear_action)
        
        self.toolbar.addSeparator()
        
        self.expand_action = QAction(qta.icon('fa5s.expand-alt', color='#9CA3AF'), "Expand/Collapse", self)
        self.expand_action.setCheckable(True)
        self.expand_action.setChecked(True)
        self.expand_action.triggered.connect(self.toggle_waveform)
        self.toolbar.addAction(self.expand_action)
        
        layout.addWidget(self.toolbar)
        
        # Create splitter for spectrogram and waveform
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setHandleWidth(4)
        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 2px;
            }
            QSplitter::handle:hover {
                background: rgba(139, 92, 246, 0.3);
            }
        """)
        
        # Create spectrogram
        self.spectrogram = OptimizedSpectrogramCanvas()
        self.splitter.addWidget(self.spectrogram)
        
        # Create waveform
        self.waveform = WaveformDisplay()
        self.splitter.addWidget(self.waveform)
        
        # Set initial sizes (70% spectrogram, 30% waveform)
        self.splitter.setSizes([700, 300])
        
        layout.addWidget(self.splitter)
        
        # Store initial splitter state
        self.waveform_visible = True
        self.saved_sizes = [700, 300]
        
    def setup_connections(self):
        """Setup signal connections"""
        # Link X axes for synchronized zoom
        self.waveform.viewbox.setXLink(self.spectrogram.viewbox)
        
        # Connect click events
        self.spectrogram.click_callback.connect(self.on_spectrogram_click)
        self.waveform.click_callback.connect(self.on_waveform_click)
        
    def set_data(self, audio_data, sample_rate, freqs, times, Sxx):
        """Set data for both displays"""
        # Set spectrogram data
        self.spectrogram.set_spectrogram_data(audio_data, sample_rate, freqs, times, Sxx)
        
        # Set waveform data
        self.waveform.set_audio_data(audio_data, sample_rate)
        
        # Setup time axis
        if hasattr(self.spectrogram, 'time_axis'):
            self.waveform.time_axis = self.spectrogram.time_axis
            
    def set_playback_position(self, position):
        """Update playback position on both displays"""
        self.spectrogram.set_playback_position(position)
        self.waveform.set_playback_position(position)
        
    def toggle_waveform(self, checked):
        """Toggle waveform visibility"""
        if checked:
            # Show waveform
            if not self.waveform_visible:
                self.splitter.setSizes(self.saved_sizes)
                self.waveform.setVisible(True)
                self.waveform_visible = True
        else:
            # Hide waveform
            if self.waveform_visible:
                self.saved_sizes = self.splitter.sizes()
                self.splitter.setSizes([1000, 0])
                self.waveform.setVisible(False)
                self.waveform_visible = False
                
    def toggle_marking(self, checked):
        """Toggle event marking mode"""
        self.event_annotator.toggle_marking(checked)
        if checked:
            self.mark_action.setIcon(qta.icon('fa5s.map-marker-alt', color='#8B5CF6'))
            self.statusBar().showMessage("Event marking enabled - click to mark start and end")
        else:
            self.mark_action.setIcon(qta.icon('fa5s.map-marker-alt', color='#9CA3AF'))
            self.statusBar().showMessage("Event marking disabled")
            
    def on_spectrogram_click(self, time, event):
        """Handle spectrogram clicks"""
        # Pass to event annotator if marking is enabled
        self.event_annotator.on_click(time, event)
        
        # Emit position for seeking
        if not self.event_annotator.marking_enabled:
            self.position_clicked.emit(time)
            
    def on_waveform_click(self, time, event):
        """Handle waveform clicks"""
        # Same as spectrogram click
        self.on_spectrogram_click(time, event)
        
    def run_detection(self):
        """Run automatic event detection"""
        if self.detector and self.spectrogram.Sxx is not None:
            # Run detection on current data
            QMessageBox.information(self, "Detection", 
                                  "Detection algorithm would run here.\n"
                                  "Configure in detector settings.")
            
    def export_annotations(self):
        """Export annotations to CSV"""
        if self.event_annotator.export_csv():
            self.statusBar().showMessage("Annotations exported successfully")
            
    def clear_annotations(self):
        """Clear all annotations"""
        reply = QMessageBox.question(self, "Clear Annotations",
                                    "Clear all annotations?",
                                    QMessageBox.StandardButton.Yes | 
                                    QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.event_annotator.clear_annotations()
            self.statusBar().showMessage("Annotations cleared")
            
    def set_annotation_metadata(self, site="", pixel="", file_start=None):
        """Set metadata for annotations"""
        self.event_annotator.set_metadata(site, pixel, file_start)
        
    def load_annotations(self, csv_path):
        """Load existing annotations from CSV"""
        self.event_annotator.set_csv_path(csv_path)
        
    def statusBar(self):
        """Get parent's status bar"""
        parent = self.parent()
        while parent:
            if hasattr(parent, 'statusBar'):
                return parent.statusBar()
            parent = parent.parent()
        return None