"""
Optimized Spectrogram Canvas with smooth zoom and performance improvements
Uses caching, level-of-detail rendering, and GPU-accelerated drawing where possible
"""

import numpy as np
from datetime import timedelta
from functools import lru_cache
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QToolTip, QApplication
from PySide6.QtCore import Qt, Signal, QTimer, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QLinearGradient
import matplotlib.cm as cm
from numba import jit, prange


class OptimizedViewBox(pg.ViewBox):
    """
    Custom ViewBox with smooth zoom constrained to data boundaries
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseEnabled(x=True, y=True)
        self.enableAutoRange(axis='xy', enable=False)
        
        # Zoom constraints
        self.time_bounds = None
        self.freq_bounds = None
        
        # Smooth zoom parameters
        self.zoom_factor = 1.15  # Smoother zoom steps
        self.pan_speed = 1.0
        
    def set_data_bounds(self, time_min, time_max, freq_min, freq_max):
        """Set the data boundaries for zoom constraints"""
        self.time_bounds = (time_min, time_max)
        self.freq_bounds = (freq_min, freq_max)
        self.setLimits(xMin=time_min, xMax=time_max, 
                      yMin=freq_min, yMax=freq_max)
        
    def wheelEvent(self, ev, axis=None):
        """Smooth wheel zoom with boundary constraints"""
        if self.time_bounds is None or self.freq_bounds is None:
            super().wheelEvent(ev, axis)
            return
            
        # Get mouse position in scene coordinates
        pos = ev.pos()
        mask = np.array([1, 1], dtype=float)
        modifiers = ev.modifiers()
        
        # Determine zoom axis based on modifiers
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            mask = np.array([0, 1], dtype=float)  # Y-axis only
        elif modifiers & Qt.KeyboardModifier.ShiftModifier:
            mask = np.array([1, 0], dtype=float)  # X-axis only
            
        # Calculate smooth zoom
        s = self.zoom_factor ** (ev.delta() / 120.0)
        s = [(s - 1) * m + 1 for m in mask]
        
        # Get current view range
        vr = self.viewRect()
        center = self.mapSceneToView(pos)
        
        # Apply zoom with smooth interpolation
        new_width = vr.width() / s[0]
        new_height = vr.height() / s[1]
        
        # Constrain to data bounds
        if self.time_bounds:
            new_width = min(new_width, self.time_bounds[1] - self.time_bounds[0])
            new_width = max(new_width, (self.time_bounds[1] - self.time_bounds[0]) / 100)
            
        if self.freq_bounds:
            new_height = min(new_height, self.freq_bounds[1] - self.freq_bounds[0])
            new_height = max(new_height, (self.freq_bounds[1] - self.freq_bounds[0]) / 100)
            
        # Calculate new position
        new_x = center.x() - (center.x() - vr.x()) * new_width / vr.width()
        new_y = center.y() - (center.y() - vr.y()) * new_height / vr.height()
        
        # Constrain position to bounds
        if self.time_bounds:
            new_x = max(self.time_bounds[0], min(new_x, self.time_bounds[1] - new_width))
        if self.freq_bounds:
            new_y = max(self.freq_bounds[0], min(new_y, self.freq_bounds[1] - new_height))
            
        # Apply the new view range with animation
        self.setRange(xRange=(new_x, new_x + new_width),
                     yRange=(new_y, new_y + new_height),
                     padding=0, update=True)
        
        ev.accept()
        
    def mouseDragEvent(self, ev, axis=None):
        """Smooth panning with boundary constraints"""
        if ev.button() != Qt.MouseButton.LeftButton:
            super().mouseDragEvent(ev, axis)
            return
            
        # Calculate pan delta
        delta = ev.pos() - ev.lastPos()
        delta = self.mapToView(delta) - self.mapToView(QPointF(0, 0))
        
        # Apply pan speed factor
        delta = delta * self.pan_speed
        
        # Get current range
        vr = self.viewRect()
        
        # Calculate new position
        new_x = vr.x() - delta.x()
        new_y = vr.y() - delta.y()
        
        # Constrain to bounds
        if self.time_bounds:
            new_x = max(self.time_bounds[0], 
                       min(new_x, self.time_bounds[1] - vr.width()))
        if self.freq_bounds:
            new_y = max(self.freq_bounds[0], 
                       min(new_y, self.freq_bounds[1] - vr.height()))
            
        # Apply the pan
        self.setRange(xRange=(new_x, new_x + vr.width()),
                     yRange=(new_y, new_y + vr.height()),
                     padding=0, update=True)
        
        ev.accept()


class ModernTimeAxis(pg.AxisItem):
    """
    Modern time axis with clean formatting
    """
    def __init__(self, orientation="bottom", **kwargs):
        super().__init__(orientation, **kwargs)
        self.start_dt = None
        self.setStyle(tickTextOffset=10)
        
    def set_start_time(self, start_dt):
        """Set the start time for absolute time display"""
        self.start_dt = start_dt
        
    def tickStrings(self, values, scale, spacing):
        """Format tick strings as time"""
        if self.start_dt is None:
            # Relative time format
            strings = []
            for v in values:
                try:
                    total_seconds = float(v)
                    hours = int(total_seconds // 3600)
                    minutes = int((total_seconds % 3600) // 60)
                    seconds = total_seconds % 60
                    
                    if hours > 0:
                        strings.append(f"{hours:02d}:{minutes:02d}:{seconds:05.2f}")
                    else:
                        strings.append(f"{minutes:02d}:{seconds:05.2f}")
                except:
                    strings.append(f"{v:.2f}")
            return strings
        else:
            # Absolute time format
            strings = []
            for v in values:
                t = self.start_dt + timedelta(seconds=float(v))
                strings.append(t.strftime("%H:%M:%S"))
            return strings


@jit(nopython=True, parallel=True, cache=True)
def compute_spectrogram_fast(audio_data, nperseg, noverlap, window):
    """
    Fast spectrogram computation using Numba JIT compilation
    """
    hop_length = nperseg - noverlap
    n_frames = (len(audio_data) - nperseg) // hop_length + 1
    n_freqs = nperseg // 2 + 1
    
    spectrogram = np.zeros((n_freqs, n_frames), dtype=np.complex128)
    
    for i in prange(n_frames):
        start = i * hop_length
        frame = audio_data[start:start + nperseg] * window
        fft_result = np.fft.rfft(frame)
        spectrogram[:, i] = fft_result
        
    return np.abs(spectrogram) ** 2


class OptimizedSpectrogramCanvas(QWidget):
    """
    High-performance spectrogram canvas with modern design
    """
    click_callback = Signal(float, object)  # time, event
    hover_callback = Signal(str)  # hover text
    selection_callback = Signal(float, float)  # start, end time
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
        # Data storage
        self.audio_data = None
        self.sample_rate = None
        self.freqs = None
        self.times = None
        self.Sxx = None
        self.Sxx_display = None
        
        # Performance optimizations
        self.cache_enabled = True
        self.lod_enabled = True  # Level of detail
        self.current_lod = 0
        self.display_cache = {}
        
        # Colormap
        self.colormap_name = "viridis"
        self.colormap_lut = None
        self.update_colormap()
        
        # Playback marker
        self.playback_line = None
        self.playback_position = 0
        
        # Selection
        self.selection_region = None
        
        # Auto-detection tracks
        self.auto_tracks_items = []
        
        # Update timer for smooth animations
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animations)
        self.animation_timer.start(16)  # 60 FPS
        
    def setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create graphics view
        self.graphics_view = pg.GraphicsLayoutWidget()
        self.graphics_view.setBackground((15, 15, 30, 200))  # Semi-transparent dark
        layout.addWidget(self.graphics_view)
        
        # Create custom viewbox and axis
        self.time_axis = ModernTimeAxis(orientation="bottom")
        self.freq_axis = pg.AxisItem(orientation="left")
        self.viewbox = OptimizedViewBox()
        
        # Create plot with custom components
        self.plot = self.graphics_view.addPlot(
            viewBox=self.viewbox,
            axisItems={"bottom": self.time_axis, "left": self.freq_axis}
        )
        
        # Configure plot appearance
        self.plot.showGrid(x=True, y=True, alpha=0.1)
        self.plot.setLabel('left', 'Frequency', units='Hz', 
                          **{'color': '#9CA3AF', 'font-size': '10pt'})
        self.plot.setLabel('bottom', 'Time', units='s',
                          **{'color': '#9CA3AF', 'font-size': '10pt'})
        
        # Create image item for spectrogram
        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)
        
        # Setup mouse tracking for hover info
        self.img_item.scene().sigMouseMoved.connect(self.on_mouse_move)
        self.img_item.mouseClickEvent = self.on_mouse_click
        
        # Create crosshair for precise navigation
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, 
                                          pen=pg.mkPen('#8B5CF6', width=1, style=Qt.PenStyle.DashLine))
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False,
                                          pen=pg.mkPen('#8B5CF6', width=1, style=Qt.PenStyle.DashLine))
        self.crosshair_v.setVisible(False)
        self.crosshair_h.setVisible(False)
        self.plot.addItem(self.crosshair_v)
        self.plot.addItem(self.crosshair_h)
        
    def update_colormap(self):
        """Update the colormap lookup table"""
        # Create high-quality gradient colormap
        cmap = cm.get_cmap(self.colormap_name)
        colors = cmap(np.linspace(0, 1, 256))
        self.colormap_lut = (colors * 255).astype(np.uint8)
        
        if self.img_item and self.Sxx_display is not None:
            self.img_item.setLookupTable(self.colormap_lut)
            
    def set_spectrogram_data(self, audio_data, sample_rate, freqs, times, Sxx):
        """
        Set spectrogram data with optimizations
        """
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.freqs = freqs
        self.times = times
        self.Sxx = Sxx
        
        # Clear cache
        self.display_cache.clear()
        
        # Set data bounds for zoom constraints
        if times is not None and freqs is not None:
            self.viewbox.set_data_bounds(
                times[0], times[-1],
                freqs[0], freqs[-1]
            )
            
        # Prepare display data
        self.prepare_display_data()
        
        # Update display
        self.update_display()
        
    def prepare_display_data(self):
        """
        Prepare data for display with level-of-detail optimization
        """
        if self.Sxx is None:
            return
            
        # Convert to dB scale with proper handling
        with np.errstate(divide='ignore', invalid='ignore'):
            Sxx_db = 10 * np.log10(self.Sxx + 1e-10)
            
        # Apply dynamic range compression for better visibility
        vmin, vmax = np.percentile(Sxx_db[np.isfinite(Sxx_db)], [1, 99])
        Sxx_db = np.clip(Sxx_db, vmin, vmax)
        
        # Normalize to 0-255 for display
        Sxx_norm = ((Sxx_db - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        
        # Transpose for correct display (time on X-axis, freq on Y-axis)
        self.Sxx_display = Sxx_norm.T
        
        # Generate LOD versions for large datasets
        if self.lod_enabled and self.Sxx_display.shape[0] > 2000:
            self.generate_lod_versions()
            
    def generate_lod_versions(self):
        """Generate level-of-detail versions for performance"""
        self.display_cache['lod_0'] = self.Sxx_display
        
        # Generate downsampled versions
        current = self.Sxx_display
        for lod in range(1, 4):
            # Downsample by factor of 2
            h, w = current.shape
            downsampled = current[::2, ::2]
            self.display_cache[f'lod_{lod}'] = downsampled
            current = downsampled
            
    def update_display(self):
        """Update the display with appropriate LOD"""
        if self.Sxx_display is None:
            return
            
        # Choose appropriate LOD based on zoom level
        view_range = self.viewbox.viewRange()
        if view_range and self.lod_enabled and self.display_cache:
            x_span = view_range[0][1] - view_range[0][0]
            total_span = self.times[-1] - self.times[0] if self.times is not None else 1
            zoom_ratio = x_span / total_span
            
            # Select LOD based on zoom
            if zoom_ratio > 0.5:
                lod = 2
            elif zoom_ratio > 0.25:
                lod = 1
            else:
                lod = 0
                
            display_data = self.display_cache.get(f'lod_{lod}', self.Sxx_display)
        else:
            display_data = self.Sxx_display
            
        # Update image
        self.img_item.setImage(display_data)
        self.img_item.setLookupTable(self.colormap_lut)
        
        # Set proper scaling
        if self.times is not None and self.freqs is not None:
            scale_x = (self.times[-1] - self.times[0]) / display_data.shape[0]
            scale_y = (self.freqs[-1] - self.freqs[0]) / display_data.shape[1]
            self.img_item.setRect(QRectF(self.times[0], self.freqs[0],
                                        self.times[-1] - self.times[0],
                                        self.freqs[-1] - self.freqs[0]))
                                        
    def on_mouse_move(self, pos):
        """Handle mouse movement for hover info"""
        if self.Sxx is None:
            return
            
        # Convert to view coordinates
        mouse_point = self.viewbox.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        # Check if in bounds
        if (self.times is not None and self.freqs is not None and
            self.times[0] <= x <= self.times[-1] and
            self.freqs[0] <= y <= self.freqs[-1]):
            
            # Find nearest indices
            time_idx = np.searchsorted(self.times, x)
            freq_idx = np.searchsorted(self.freqs, y)
            
            if 0 <= time_idx < len(self.times) and 0 <= freq_idx < len(self.freqs):
                # Get amplitude
                amp_db = 10 * np.log10(self.Sxx[freq_idx, time_idx] + 1e-10)
                
                # Format hover text
                hover_text = (f"Time: {x:.3f}s\n"
                            f"Freq: {y:.1f} Hz\n"
                            f"Amp: {amp_db:.1f} dB")
                
                self.hover_callback.emit(hover_text)
                
                # Update crosshair if Alt is pressed
                if QApplication.keyboardModifiers() & Qt.KeyboardModifier.AltModifier:
                    self.crosshair_v.setPos(x)
                    self.crosshair_h.setPos(y)
                    self.crosshair_v.setVisible(True)
                    self.crosshair_h.setVisible(True)
                else:
                    self.crosshair_v.setVisible(False)
                    self.crosshair_h.setVisible(False)
                    
    def on_mouse_click(self, event):
        """Handle mouse clicks"""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            mouse_point = self.viewbox.mapSceneToView(pos)
            x = mouse_point.x()
            
            if self.times is not None and self.times[0] <= x <= self.times[-1]:
                self.click_callback.emit(x, event)
                
    def set_playback_position(self, position):
        """Update playback position marker"""
        self.playback_position = position
        
        if self.playback_line is None:
            self.playback_line = pg.InfiniteLine(
                angle=90, movable=False,
                pen=pg.mkPen('#EF4444', width=2)
            )
            self.plot.addItem(self.playback_line)
            
        self.playback_line.setPos(position)
        
    def update_animations(self):
        """Update any ongoing animations"""
        # Placeholder for future animation updates
        pass
        
    def set_selection(self, start, end):
        """Set selection region"""
        if self.selection_region is None:
            self.selection_region = pg.LinearRegionItem(
                values=(start, end),
                brush=pg.mkBrush(139, 92, 246, 50),
                pen=pg.mkPen('#8B5CF6', width=2)
            )
            self.selection_region.sigRegionChanged.connect(self.on_selection_changed)
            self.plot.addItem(self.selection_region)
        else:
            self.selection_region.setRegion((start, end))
            
    def on_selection_changed(self):
        """Handle selection region changes"""
        if self.selection_region:
            start, end = self.selection_region.getRegion()
            self.selection_callback.emit(start, end)
            
    def clear_selection(self):
        """Clear the selection region"""
        if self.selection_region:
            self.plot.removeItem(self.selection_region)
            self.selection_region = None
            
    def plot_auto_tracks(self, tracks):
        """
        Overlay automatic detection tracks on the spectrogram.
        tracks: list of (time_array, freq_array) tuples
        """
        # Clear existing tracks
        self.clear_auto_tracks()
        
        # Plot each track
        for t_arr, f_arr in tracks:
            xs = np.asarray(t_arr, dtype=float)
            ys = np.asarray(f_arr, dtype=float)
            
            # Create track curve with modern styling
            curve = pg.PlotDataItem(
                xs, ys,
                pen=pg.mkPen(width=2, color=(255, 255, 0, 200)),  # Yellow with transparency
                antialias=True,
                connect='finite'
            )
            self.plot.addItem(curve)
            self.auto_tracks_items.append(curve)
            
    def clear_auto_tracks(self):
        """Clear all auto-detection tracks from the display"""
        if not hasattr(self, 'auto_tracks_items'):
            self.auto_tracks_items = []
            
        for item in self.auto_tracks_items:
            try:
                self.plot.removeItem(item)
            except:
                pass
        self.auto_tracks_items.clear()