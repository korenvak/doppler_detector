# personal/Koren/spectrogram_gui/gui/spectrogram_canvas.py

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QToolTip, QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTransform
import pyqtgraph as pg
import numpy as np
import matplotlib.cm as cm
from datetime import timedelta

from spectrogram_gui.gui.range_selector import RangeSelector


class AxisViewBox(pg.ViewBox):
    """
    Custom ViewBox that only zooms/pans horizontally except when Ctrl is held.
    When Ctrl is held, zoom/pan vertically only. Likewise for Shift → vertical pan.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseEnabled(x=True, y=False)

    def wheelEvent(self, ev, axis=None):
        modifiers = ev.modifiers()
        if modifiers & Qt.ControlModifier:
            self.setMouseEnabled(x=False, y=True)
        else:
            self.setMouseEnabled(x=True, y=False)
        super().wheelEvent(ev, axis)
        self.setMouseEnabled(x=True, y=False)

    def mouseDragEvent(self, ev, axis=None):
        modifiers = ev.modifiers()
        if modifiers & Qt.ShiftModifier:
            self.setMouseEnabled(x=False, y=True)
        else:
            self.setMouseEnabled(x=True, y=False)
        super().mouseDragEvent(ev, axis)
        self.setMouseEnabled(x=True, y=False)

    def keyPressEvent(self, ev):
        # add quick zoom shortcuts: +/- to zoom X, 0 to reset
        if ev.key() in (Qt.Key_Plus, Qt.Key_Equal):
            xr, yr = self.viewRange()
            cx = 0.5 * (xr[0] + xr[1])
            span = 0.5 * (xr[1] - xr[0]) * 0.8
            self.setXRange(cx - span, cx + span, padding=0)
        elif ev.key() in (Qt.Key_Minus, Qt.Key_Underscore):
            xr, yr = self.viewRange()
            cx = 0.5 * (xr[0] + xr[1])
            span = 0.5 * (xr[1] - xr[0]) / 0.8
            self.setXRange(cx - span, cx + span, padding=0)
        elif ev.key() == Qt.Key_0:
            self.autoRange()
        else:
            super().keyPressEvent(ev)


class TimeAxisItem(pg.AxisItem):
    """
    AxisItem that displays times as YYYY-MM-DD HH:MM:SS based on filename start time.
    """
    def __init__(self, orientation="bottom", **kwargs):
        super().__init__(orientation, **kwargs)
        self.start_dt = None
        self.end_dt = None

    def set_start_time(self, start_dt, end_dt=None):
        """
        Store the datetime corresponding to times[0] on the X axis.
        """
        self.start_dt = start_dt
        self.end_dt = end_dt

    def tickStrings(self, values, scale, spacing):
        # if no start_dt or values aren't plain floats, fall back to relative time display
        if self.start_dt is None or not all(np.isscalar(v) for v in values):
            # Show as HH:MM:SS relative time format instead of raw seconds
            out = []
            for v in values:
                try:
                    total_seconds = float(v)
                    hours = int(total_seconds // 3600)
                    minutes = int((total_seconds % 3600) // 60)
                    seconds = int(total_seconds % 60)
                    out.append(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
                except:
                    out.append(f"{v:.2f}")
            return out

        out = []
        for v in values:
            # v is in seconds since the start of the file
            t = self.start_dt + timedelta(seconds=float(v))
            # Show only HH:MM:SS on the axis (date goes in CSV only)
            out.append(t.strftime("%H:%M:%S"))
        return out


class SpectrogramCanvas(QWidget):
    """
    Displays the spectrogram as an ImageItem. Supports:
      - Hover info callback (time/freq/amp) via hover_callback
      - ALT-held crosshair + labels
      - Range selection via RangeSelector (right-click + drag)
      - Click to seek or annotate via click_callback
      - Playback position marker (vertical red line)
    """
    click_callback = None   # expects function(rel_sec, event)
    hover_callback = None   # expects function(text)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.view = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.view)

        # Create custom ViewBox + bottom axis
        self.axis = TimeAxisItem(orientation="bottom")
        self.vb = AxisViewBox()
        self.plot = self.view.addPlot(viewBox=self.vb, axisItems={"bottom": self.axis})
        self.plot.showGrid(x=True, y=True)
        self.plot.setLabel('left', 'Frequency', units='Hz')
        self.plot.setLabel('bottom', 'Time')

        # ImageItem to hold Sxx
        self.img_item = pg.ImageItem()
        self.img_item.setOpacity(0.9)
        self.plot.addItem(self.img_item)

        # Data placeholders
        self.freqs = None
        self.times = None
        self.Sxx = None         # transposed for display
        self.Sxx_raw = None     # original (freq × time)
        self.current_sxx = None  # ← NEW: for FilterDialog undo/redo
        self.start_time = None
        self.colormap_name = "gray"

        # ALT-held crosshairs + text
        self.alt_lines = []
        self.alt_texts = []

        # Persist last view range for zoom persistence
        self.view_state = None
        self.vb.sigRangeChanged.connect(self._store_view_state)

        # Range selection
        self.range_selector = RangeSelector(self.plot)
        self.range_selector.range_changed.connect(self.on_range_selected)
        self.selected_range = None  # tuple (t_start, t_end)

        # Connect mouse events
        self.plot.scene().sigMouseClicked.connect(self._on_click)
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_move)

        # Playback position (vertical red line)
        self.playback_line = pg.InfiniteLine(angle=90, pen=pg.mkPen('r', width=1))
        self.playback_line.hide()
        self.plot.addItem(self.playback_line)

        # Default spectrogram parameters
        self.spectrogram_params = {
            "window": "hann",
            "nperseg": 1024,
            "noverlap": 512
        }
        self.auto_tracks_items = []

    def _store_view_state(self, *_, **__):
        try:
            self.view_state = self.vb.viewRange()
        except Exception:
            self.view_state = None

    def clear_auto_tracks(self):
        print("[Canvas] clear_auto_tracks()")
        for item in self.auto_tracks_items:
            try:
                self.plot.removeItem(item)
            except:
                pass
        self.auto_tracks_items = []

    def remove_items(self, items):
        for item in items:
            try:
                self.plot.removeItem(item)
            except Exception:
                pass

    def plot_auto_tracks(self, tracks):
        """
        Overlay automatic detection tracks on the spectrogram.
        Each track is provided as a tuple of (times_array, freqs_array).
        """
        for t_arr, f_arr in tracks:
            xs = np.asarray(t_arr, dtype=float)
            ys = np.asarray(f_arr, dtype=float)
            curve = pg.PlotDataItem(
                xs, ys,
                pen=pg.mkPen(width=2, color=(255, 255, 0)),
                antialias=True,
                connect='finite'
            )
            self.plot.addItem(curve)
            self.auto_tracks_items.append(curve)

    def plot_spectrogram(self, freqs, times, Sxx_raw, start_time, end_time=None, maintain_view=False):
        """
        Plot the spectrogram image (freqs × times × Sxx_raw),
        then clear any old ALT crosshairs, range selections,
        auto-tracks, and snapshot current arrays for filters/undo.
        """
        # --- 0) Wipe out any existing auto-detect tracks immediately
        old_view = self.view_state if maintain_view else None
        self.clear_auto_tracks()

        # --- 1) Store the data + snapshots for filters & undo
        self.freqs         = freqs
        self.times         = times
        self.current_freqs = freqs.copy()
        self.current_times = times.copy()
        self.Sxx_raw       = Sxx_raw
        self.current_sxx   = Sxx_raw.copy()
        # transpose for display
        self.Sxx           = Sxx_raw.T

        # --- 2) Set the reference start and end times (for TimeAxisItem & hover)
        self.start_time = start_time
        self.end_time = end_time
        self.axis.set_start_time(start_time, end_time)

        # --- 3) Compute display levels and LUT
        levels = (np.min(self.Sxx), np.max(self.Sxx))
        lut    = self.get_colormap_lut(self.colormap_name)

        # --- 4) Draw the image
        self.img_item.setImage(self.Sxx, autoLevels=False, levels=levels, lut=lut)

        # --- 5) Scale pixel space to real time/freq axes
        dt = times[1] - times[0] if len(times) > 1 else 1.0
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        self.img_item.resetTransform()
        transform = QTransform()
        transform.scale(dt, df)
        self.img_item.setTransform(transform)
        self.img_item.setPos(times[0], freqs[0])

        # --- 6) Constrain zoom/pan to data bounds
        self.vb.setLimits(
            xMin=times[0], xMax=times[-1],
            yMin=freqs[0], yMax=freqs[-1]
        )

        # --- 7) Auto-range one shot or restore previous view
        if old_view is not None:
            xr = [max(times[0], old_view[0][0]), min(times[-1], old_view[0][1])]
            yr = [max(freqs[0], old_view[1][0]), min(freqs[-1], old_view[1][1])]
            self.vb.setRange(xRange=xr, yRange=yr, padding=0)
        else:
            self.vb.autoRange()

        # --- 8) Remove any old ALT-crosshairs + labels
        for item in self.alt_lines + self.alt_texts:
            self.plot.removeItem(item)
        self.alt_lines.clear()
        self.alt_texts.clear()

        # --- 9) Reset any range selection & hide playback marker
        self.selected_range = None
        self.range_selector.clear()
        self.playback_line.hide()




    def set_colormap(self, cmap_name):
        """
        Change colormap and re-plot if we already have data.
        """
        valid = ["gray", "viridis", "magma", "inferno", "plasma"]
        if cmap_name not in valid:
            return
        self.colormap_name = cmap_name
        if self.freqs is not None and self.times is not None and self.Sxx_raw is not None:
            self.plot_spectrogram(self.freqs, self.times, self.Sxx_raw, self.start_time, self.end_time, maintain_view=True)

    def get_colormap_lut(self, cmap_name):
        """
        Return a 256×4 numpy array LUT from matplotlib.
        """
        try:
            cmap = cm.get_cmap(cmap_name)
            lut = (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
            return lut
        except Exception:
            return None

    def on_range_selected(self, t_start, t_end):
        self.selected_range = (t_start, t_end)

    def _on_mouse_move(self, pos):
        mouse_point = self.plot.vb.mapSceneToView(pos)
        x_time = float(mouse_point.x())
        y_freq = float(mouse_point.y())

        if self.times is None or self.freqs is None or self.Sxx_raw is None:
            return

        time_idx = np.argmin(np.abs(self.times - x_time))
        freq_idx = np.argmin(np.abs(self.freqs - y_freq))
        time_idx = max(0, min(time_idx, len(self.times) - 1))
        freq_idx = max(0, min(freq_idx, len(self.freqs) - 1))

        amp_db = self.Sxx_raw[freq_idx, time_idx]
        freq_hz = self.freqs[freq_idx]

        if self.start_time:
            actual_time = self.start_time + timedelta(seconds=float(self.times[time_idx]))
            time_str = actual_time.strftime("%H:%M:%S")
        else:
            time_str = f"{self.times[time_idx]:.3f}s"

        text = (
            f"Time: {time_str}   "
            f"Freq: {freq_hz:.1f} Hz   "
            f"Amp: {amp_db:.1f} dB"
        )
        if self.hover_callback:
            self.hover_callback(text)

        modifiers = QApplication.keyboardModifiers()
        if modifiers & Qt.AltModifier:
            for item in self.alt_lines + self.alt_texts:
                self.plot.removeItem(item)
            self.alt_lines.clear()
            self.alt_texts.clear()

            v_line = pg.PlotDataItem(
                [x_time, x_time], [self.freqs[0], y_freq], pen=pg.mkPen('y', width=1)
            )
            h_line = pg.PlotDataItem(
                [self.times[0], x_time], [y_freq, y_freq], pen=pg.mkPen('y', width=1)
            )
            self.plot.addItem(v_line); self.plot.addItem(h_line)
            self.alt_lines.extend([v_line, h_line])

            time_label = pg.TextItem(time_str, anchor=(0.5, 1.0), color='y')
            time_label.setPos(x_time, self.freqs[0])
            freq_label = pg.TextItem(f"{freq_hz:.1f} Hz", anchor=(1.0, 0.5), color='y')
            x_left = self.vb.viewRect().left() if self.vb.viewRect() else self.times[0]
            freq_label.setPos(x_left, y_freq)
            self.plot.addItem(time_label); self.plot.addItem(freq_label)
            self.alt_texts.extend([time_label, freq_label])
        else:
            if self.alt_lines or self.alt_texts:
                for item in self.alt_lines + self.alt_texts:
                    self.plot.removeItem(item)
                self.alt_lines.clear()
                self.alt_texts.clear()

    def _on_click(self, event):
        pos = event.scenePos()
        mouse_point = self.plot.vb.mapSceneToView(pos)
        x_time = float(mouse_point.x())
        y_freq = float(mouse_point.y())

        if self.times is None or self.freqs is None or self.Sxx_raw is None:
            return

        if event.button() == Qt.RightButton:
            time_idx = np.argmin(np.abs(self.times - x_time))
            freq_idx = np.argmin(np.abs(self.freqs - y_freq))
            time_idx = max(0, min(time_idx, len(self.times) - 1))
            freq_idx = max(0, min(freq_idx, len(self.freqs) - 1))

            amp_db = self.Sxx_raw[freq_idx, time_idx]
            freq_hz = self.freqs[freq_idx]
            if self.start_time:
                actual_time = self.start_time + timedelta(seconds=self.times[time_idx])
                time_str = actual_time.strftime('%H:%M:%S')
            else:
                time_str = f"{self.times[time_idx]:.3f}s"

            tip_text = (
                f"Time: {time_str}\n"
                f"Freq: {freq_hz:.1f} Hz\n"
                f"Amp: {amp_db:.1f} dB"
            )
            QToolTip.showText(event.screenPos().toPoint(), tip_text)
            return

        if self.click_callback:
            rel_sec = x_time - self.times[0]
            self.click_callback(rel_sec, event)

    def clear_annotations(self):
        for item in list(self.plot.items):
            if (
                isinstance(item, (pg.TextItem, pg.InfiniteLine))
                and item not in (self.img_item, self.playback_line)
            ):
                if QApplication.keyboardModifiers() & Qt.AltModifier:
                    if item in self.alt_lines or item in self.alt_texts:
                        continue
                self.plot.removeItem(item)

    def update_playback_position(self, ms):
        if self.times is None:
            return
        rel_sec = ms / 1000.0
        x = self.times[0] + rel_sec
        self.playback_line.setValue(x)
        self.playback_line.show()

    def set_start_time(self, start_dt, end_dt=None):
        self.start_time = start_dt
        self.end_time = end_dt
        self.axis.set_start_time(start_dt, end_dt)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if hasattr(self, 'vb'):
            if self.view_state is not None:
                self.vb.setRange(xRange=self.view_state[0], yRange=self.view_state[1], padding=0)
            else:
                self.vb.autoRange()
