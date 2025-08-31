"""
Enhanced Spectrogram Canvas with modern features:
- Smooth zoom with mouse wheel (Ctrl for Y-axis)
- Pan with middle mouse button
- Normalization options
- Colormap selection
- Grid toggle
- Measurement tools
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QSlider, QLabel, QComboBox, QCheckBox, QToolButton,
    QButtonGroup, QMenu, QAction, QWidgetAction, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QPointF, QRectF
from PyQt5.QtGui import QIcon, QCursor, QPen, QColor, QFont
import pyqtgraph as pg
import numpy as np
import matplotlib.cm as cm
from datetime import timedelta
import qtawesome as qta

from spectrogram_gui.gui.range_selector import RangeSelector


class ModernViewBox(pg.ViewBox):
    """Enhanced ViewBox with smooth zoom and pan controls"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseEnabled(x=True, y=True)
        self.zoom_factor = 1.2
        self.pan_speed = 1.0
        
    def wheelEvent(self, ev, axis=None):
        """Enhanced wheel zoom with smooth factor"""
        modifiers = ev.modifiers()
        
        # Get zoom center point
        pos = ev.pos()
        
        # Calculate zoom factor
        if ev.delta() > 0:
            scale_factor = self.zoom_factor
        else:
            scale_factor = 1 / self.zoom_factor
            
        # Apply zoom
        if modifiers & Qt.ControlModifier:
            # Zoom Y-axis only
            self.scaleBy((1, scale_factor), center=pos)
        elif modifiers & Qt.ShiftModifier:
            # Zoom X-axis only  
            self.scaleBy((scale_factor, 1), center=pos)
        else:
            # Zoom both axes
            self.scaleBy((scale_factor, scale_factor), center=pos)
            
        ev.accept()
        
    def mouseDragEvent(self, ev, axis=None):
        """Enhanced pan with middle mouse button"""
        if ev.button() == Qt.MiddleButton:
            if ev.isStart():
                self.pan_start = ev.pos()
            elif ev.isFinish():
                self.pan_start = None
            else:
                # Calculate pan delta
                delta = ev.pos() - ev.lastPos()
                self.translateBy(delta * self.pan_speed)
        else:
            super().mouseDragEvent(ev, axis)


class TimeAxisItem(pg.AxisItem):
    """Enhanced time axis with better formatting"""
    
    def __init__(self, orientation="bottom", **kwargs):
        super().__init__(orientation, **kwargs)
        self.start_dt = None
        self.setStyle(tickTextOffset=10)
        
    def set_start_time(self, dt):
        self.start_dt = dt
        
    def tickStrings(self, values, scale, spacing):
        if self.start_dt is None:
            return [f"{v:.2f}s" for v in values]
            
        strings = []
        for v in values:
            t = self.start_dt + timedelta(seconds=float(v))
            if spacing < 1:
                # Show milliseconds
                strings.append(t.strftime("%H:%M:%S.%f")[:-3])
            elif spacing < 60:
                # Show seconds
                strings.append(t.strftime("%H:%M:%S"))
            else:
                # Show minutes
                strings.append(t.strftime("%H:%M"))
        return strings


class EnhancedSpectrogramCanvas(QWidget):
    """Modern spectrogram canvas with enhanced features"""
    
    click_callback = None
    hover_callback = None
    range_selected = pyqtSignal(tuple)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.init_data()
        self.setup_interactions()
        
    def setup_ui(self):
        """Setup the modern UI layout"""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Create toolbar
        self.toolbar = self.create_toolbar()
        self.layout.addWidget(self.toolbar)
        
        # Create graphics view
        self.view = pg.GraphicsLayoutWidget()
        self.view.setBackground('#141824')
        self.layout.addWidget(self.view)
        
        # Create custom ViewBox and plot
        self.axis = TimeAxisItem(orientation="bottom")
        self.vb = ModernViewBox()
        self.plot = self.view.addPlot(viewBox=self.vb, axisItems={"bottom": self.axis})
        
        # Style the plot
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel('left', 'Frequency', units='Hz', color='#F9FAFB')
        self.plot.setLabel('bottom', 'Time', color='#F9FAFB')
        
        # Set axis colors
        for axis in ['left', 'bottom']:
            self.plot.getAxis(axis).setPen(pg.mkPen('#374151', width=1))
            self.plot.getAxis(axis).setTextPen('#F9FAFB')
            
        # Create image item for spectrogram
        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)
        
        # Create colorbar
        self.colorbar = pg.ColorBarItem(
            values=(0, 1),
            colorMap='viridis',
            width=15,
            interactive=False
        )
        self.colorbar.setImageItem(self.img_item)
        
        # Measurement tools
        self.crosshair_lines = []
        self.measurement_lines = []
        self.annotations = []
        
        # Playback position line
        self.playback_line = pg.InfiniteLine(
            angle=90, 
            pen=pg.mkPen('#EF4444', width=2, style=Qt.DashLine)
        )
        self.playback_line.hide()
        self.plot.addItem(self.playback_line)
        
        # Range selector
        self.range_selector = RangeSelector(self.plot)
        self.range_selector.range_changed.connect(self.on_range_selected)
        
    def create_toolbar(self):
        """Create modern toolbar with controls"""
        toolbar = QFrame()
        toolbar.setObjectName("modernCard")
        toolbar.setMaximumHeight(50)
        
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        
        # Zoom controls
        zoom_group = QFrame()
        zoom_layout = QHBoxLayout(zoom_group)
        zoom_layout.setContentsMargins(0, 0, 0, 0)
        zoom_layout.setSpacing(4)
        
        self.zoom_in_btn = QToolButton()
        self.zoom_in_btn.setIcon(qta.icon('mdi.magnify-plus', color='#F9FAFB'))
        self.zoom_in_btn.setToolTip("Zoom In")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(self.zoom_in_btn)
        
        self.zoom_out_btn = QToolButton()
        self.zoom_out_btn.setIcon(qta.icon('mdi.magnify-minus', color='#F9FAFB'))
        self.zoom_out_btn.setToolTip("Zoom Out")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(self.zoom_out_btn)
        
        self.zoom_reset_btn = QToolButton()
        self.zoom_reset_btn.setIcon(qta.icon('mdi.fullscreen', color='#F9FAFB'))
        self.zoom_reset_btn.setToolTip("Reset Zoom")
        self.zoom_reset_btn.clicked.connect(self.zoom_reset)
        zoom_layout.addWidget(self.zoom_reset_btn)
        
        layout.addWidget(zoom_group)
        
        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("background: #374151;")
        layout.addWidget(sep1)
        
        # Normalization controls
        norm_group = QFrame()
        norm_layout = QHBoxLayout(norm_group)
        norm_layout.setContentsMargins(0, 0, 0, 0)
        norm_layout.setSpacing(4)
        
        norm_label = QLabel("Normalize:")
        norm_label.setStyleSheet("color: #9CA3AF;")
        norm_layout.addWidget(norm_label)
        
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["None", "Min-Max", "Percentile", "Log Scale", "dB Scale"])
        self.norm_combo.setMaximumWidth(120)
        self.norm_combo.currentTextChanged.connect(self.apply_normalization)
        norm_layout.addWidget(self.norm_combo)
        
        layout.addWidget(norm_group)
        
        # Colormap selector
        cmap_group = QFrame()
        cmap_layout = QHBoxLayout(cmap_group)
        cmap_layout.setContentsMargins(0, 0, 0, 0)
        cmap_layout.setSpacing(4)
        
        cmap_label = QLabel("Colormap:")
        cmap_label.setStyleSheet("color: #9CA3AF;")
        cmap_layout.addWidget(cmap_label)
        
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems([
            "viridis", "plasma", "inferno", "magma",
            "jet", "turbo", "hot", "cool", "gray"
        ])
        self.cmap_combo.setMaximumWidth(100)
        self.cmap_combo.currentTextChanged.connect(self.change_colormap)
        cmap_layout.addWidget(self.cmap_combo)
        
        layout.addWidget(cmap_group)
        
        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("background: #374151;")
        layout.addWidget(sep2)
        
        # Tools
        tools_group = QFrame()
        tools_layout = QHBoxLayout(tools_group)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(4)
        
        self.crosshair_btn = QToolButton()
        self.crosshair_btn.setIcon(qta.icon('mdi.crosshairs', color='#F9FAFB'))
        self.crosshair_btn.setToolTip("Crosshair")
        self.crosshair_btn.setCheckable(True)
        self.crosshair_btn.toggled.connect(self.toggle_crosshair)
        tools_layout.addWidget(self.crosshair_btn)
        
        self.measure_btn = QToolButton()
        self.measure_btn.setIcon(qta.icon('mdi.ruler', color='#F9FAFB'))
        self.measure_btn.setToolTip("Measure")
        self.measure_btn.setCheckable(True)
        self.measure_btn.toggled.connect(self.toggle_measure)
        tools_layout.addWidget(self.measure_btn)
        
        self.grid_btn = QToolButton()
        self.grid_btn.setIcon(qta.icon('mdi.grid', color='#F9FAFB'))
        self.grid_btn.setToolTip("Toggle Grid")
        self.grid_btn.setCheckable(True)
        self.grid_btn.setChecked(True)
        self.grid_btn.toggled.connect(self.toggle_grid)
        tools_layout.addWidget(self.grid_btn)
        
        layout.addWidget(tools_group)
        
        # Stretch
        layout.addStretch()
        
        # Info label
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #9CA3AF; font-size: 14px;")
        layout.addWidget(self.info_label)
        
        return toolbar
        
    def init_data(self):
        """Initialize data structures"""
        self.freqs = None
        self.times = None
        self.Sxx = None
        self.Sxx_raw = None
        self.Sxx_normalized = None
        self.start_time = None
        self.colormap_name = "viridis"
        self.selected_range = None
        
        # View state for persistence
        self.view_state = None
        self.vb.sigRangeChanged.connect(self._store_view_state)
        
        # Spectrogram parameters
        self.spectrogram_params = {
            "window": "hann",
            "nperseg": 1024,
            "noverlap": 512
        }
        
        # Tracking items
        self.auto_tracks_items = []
        self.is_crosshair_active = False
        self.is_measure_active = False
        
    def setup_interactions(self):
        """Setup mouse and keyboard interactions"""
        self.plot.scene().sigMouseClicked.connect(self._on_click)
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_move)
        
    def set_spectrogram(self, freqs, times, Sxx, start_time=None):
        """Set spectrogram data with enhanced processing"""
        self.freqs = freqs
        self.times = times
        self.Sxx_raw = Sxx.copy()
        self.Sxx = Sxx.T  # Transpose for display
        self.start_time = start_time
        
        # Apply current normalization
        self.apply_normalization(self.norm_combo.currentText())
        
        # Set time axis
        if start_time:
            self.axis.set_start_time(start_time)
            
        # Update image
        self.update_display()
        
    def update_display(self):
        """Update the spectrogram display"""
        if self.Sxx is None:
            return
            
        # Apply colormap
        self.change_colormap(self.cmap_combo.currentText())
        
        # Set image data
        self.img_item.setImage(
            self.Sxx,
            autoLevels=False,
            levels=(np.min(self.Sxx), np.max(self.Sxx))
        )
        
        # Set image position and scale
        if self.times is not None and self.freqs is not None:
            self.img_item.setRect(QRectF(
                self.times[0], self.freqs[0],
                self.times[-1] - self.times[0],
                self.freqs[-1] - self.freqs[0]
            ))
            
        # Update colorbar
        self.colorbar.setLevels((np.min(self.Sxx), np.max(self.Sxx)))
        
        # Update info
        self.update_info()
        
    def apply_normalization(self, norm_type):
        """Apply normalization to spectrogram"""
        if self.Sxx_raw is None:
            return
            
        if norm_type == "None":
            self.Sxx = self.Sxx_raw.T
        elif norm_type == "Min-Max":
            min_val = np.min(self.Sxx_raw)
            max_val = np.max(self.Sxx_raw)
            if max_val > min_val:
                self.Sxx = ((self.Sxx_raw - min_val) / (max_val - min_val)).T
            else:
                self.Sxx = self.Sxx_raw.T
        elif norm_type == "Percentile":
            p1, p99 = np.percentile(self.Sxx_raw, [1, 99])
            clipped = np.clip(self.Sxx_raw, p1, p99)
            if p99 > p1:
                self.Sxx = ((clipped - p1) / (p99 - p1)).T
            else:
                self.Sxx = self.Sxx_raw.T
        elif norm_type == "Log Scale":
            # Add small epsilon to avoid log(0)
            log_data = np.log10(self.Sxx_raw + 1e-10)
            self.Sxx = log_data.T
        elif norm_type == "dB Scale":
            # Convert to dB
            db_data = 10 * np.log10(self.Sxx_raw + 1e-10)
            self.Sxx = db_data.T
            
        self.update_display()
        
    def change_colormap(self, cmap_name):
        """Change the colormap"""
        self.colormap_name = cmap_name
        
        # Get matplotlib colormap
        try:
            cmap = cm.get_cmap(cmap_name)
            # Convert to pyqtgraph format
            colors = cmap(np.linspace(0, 1, 256))
            colors = (colors * 255).astype(np.uint8)
            
            # Create LUT
            lut = colors[:, :3]  # Remove alpha channel
            self.img_item.setLookupTable(lut)
            
            # Update colorbar
            pg_cmap = pg.ColorMap(
                pos=np.linspace(0, 1, len(lut)),
                color=lut
            )
            self.colorbar.setColorMap(pg_cmap)
        except:
            pass
            
    def zoom_in(self):
        """Zoom in by fixed factor"""
        self.vb.scaleBy((0.8, 0.8))
        
    def zoom_out(self):
        """Zoom out by fixed factor"""
        self.vb.scaleBy((1.25, 1.25))
        
    def zoom_reset(self):
        """Reset zoom to show all data"""
        if self.times is not None and self.freqs is not None:
            self.vb.setRange(
                xRange=(self.times[0], self.times[-1]),
                yRange=(self.freqs[0], self.freqs[-1]),
                padding=0.02
            )
            
    def toggle_grid(self, checked):
        """Toggle grid visibility"""
        self.plot.showGrid(x=checked, y=checked, alpha=0.3)
        
    def toggle_crosshair(self, checked):
        """Toggle crosshair mode"""
        self.is_crosshair_active = checked
        if not checked:
            self.clear_crosshair()
            
    def toggle_measure(self, checked):
        """Toggle measurement mode"""
        self.is_measure_active = checked
        if not checked:
            self.clear_measurements()
            
    def clear_crosshair(self):
        """Clear crosshair lines"""
        for line in self.crosshair_lines:
            self.plot.removeItem(line)
        self.crosshair_lines = []
        
    def clear_measurements(self):
        """Clear measurement lines"""
        for item in self.measurement_lines:
            self.plot.removeItem(item)
        self.measurement_lines = []
        
    def update_info(self):
        """Update info label"""
        if self.Sxx is not None:
            shape = self.Sxx.shape
            if self.times is not None:
                duration = self.times[-1] - self.times[0]
                self.info_label.setText(
                    f"Size: {shape[0]}×{shape[1]} | "
                    f"Duration: {duration:.2f}s | "
                    f"Freq: {self.freqs[0]:.1f}-{self.freqs[-1]:.1f} Hz"
                )
                
    def _on_mouse_move(self, pos):
        """Handle mouse move for crosshair and hover info"""
        mouse_point = self.plot.vb.mapSceneToView(pos)
        
        if self.is_crosshair_active:
            # Update crosshair
            self.clear_crosshair()
            
            v_line = pg.InfiniteLine(
                pos=mouse_point.x(),
                angle=90,
                pen=pg.mkPen('#34D399', width=1, style=Qt.DashLine)
            )
            h_line = pg.InfiniteLine(
                pos=mouse_point.y(),
                angle=0,
                pen=pg.mkPen('#34D399', width=1, style=Qt.DashLine)
            )
            
            self.plot.addItem(v_line)
            self.plot.addItem(h_line)
            self.crosshair_lines = [v_line, h_line]
            
        # Update hover info
        if self.hover_callback and self.times is not None and self.freqs is not None:
            x, y = mouse_point.x(), mouse_point.y()
            
            # Find nearest time and frequency
            if (self.times[0] <= x <= self.times[-1] and 
                self.freqs[0] <= y <= self.freqs[-1]):
                
                time_idx = np.searchsorted(self.times, x)
                freq_idx = np.searchsorted(self.freqs, y)
                
                if 0 <= time_idx < len(self.times) and 0 <= freq_idx < len(self.freqs):
                    time_val = self.times[time_idx]
                    freq_val = self.freqs[freq_idx]
                    
                    if self.Sxx_raw is not None:
                        amp_val = self.Sxx_raw[freq_idx, time_idx]
                        info_text = f"Time: {time_val:.3f}s | Freq: {freq_val:.1f} Hz | Amp: {amp_val:.2e}"
                        self.hover_callback(info_text)
                        
    def _on_click(self, event):
        """Handle mouse clicks"""
        if not event.double():
            pos = event.scenePos()
            mouse_point = self.plot.vb.mapSceneToView(pos)
            
            if self.click_callback and self.times is not None:
                # Calculate relative time
                rel_sec = mouse_point.x() - self.times[0]
                if 0 <= rel_sec <= (self.times[-1] - self.times[0]):
                    self.click_callback(rel_sec, event)
                    
    def on_range_selected(self, range_tuple):
        """Handle range selection"""
        self.selected_range = range_tuple
        self.range_selected.emit(range_tuple)
        
    def set_playback_position(self, rel_sec):
        """Update playback position marker"""
        if self.times is not None:
            pos = self.times[0] + rel_sec
            self.playback_line.setPos(pos)
            self.playback_line.show()
            
    def hide_playback_position(self):
        """Hide playback position marker"""
        self.playback_line.hide()
        
    def _store_view_state(self):
        """Store current view state for persistence"""
        try:
            self.view_state = self.vb.viewRange()
        except:
            self.view_state = None
            
    def restore_view_state(self):
        """Restore previous view state"""
        if self.view_state:
            try:
                self.vb.setRange(
                    xRange=self.view_state[0],
                    yRange=self.view_state[1],
                    padding=0
                )
            except:
                pass