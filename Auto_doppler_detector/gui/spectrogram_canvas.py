import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import date2num, num2date, DateFormatter
from datetime import timedelta

class SpectrogramCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=4, dpi=100):
        """
        A Matplotlib canvas embedded in Qt for drawing spectrograms
        and overlaid tracks. Adds a tooltip-like info text in the top-right
        showing Time, Frequency, and Intensity under the cursor.
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self.image = None
        self.track_lines = []

        # Will store the last drawn spectrogram data
        self.freqs = None
        self.times = None
        self.Sxx_norm = None
        self.start_time = None  # datetime or None

        self.fig.tight_layout()

        # Info text in top-right corner (axes coordinates)
        self.info_text = self.ax.text(
            0.98, 0.98, "", transform=self.ax.transAxes,
            ha='right', va='top',
            color='white', fontsize=8,
            backgroundcolor='black', alpha=0.6
        )

        # Connect mouse events
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

    def plot_spectrogram(self, freqs, times, Sxx_norm, start_time=None):
        """
        Plot the normalized spectrogram and store all data for mouse-over info.
        If start_time is provided, X-axis shows absolute times; otherwise, seconds.
        """
        self.ax.clear()

        # Store for later lookups
        self.freqs = freqs
        self.times = times
        self.Sxx_norm = Sxx_norm
        self.start_time = start_time

        if start_time is not None:
            # Convert each time-bin t to a matplotlib date number
            dt_nums = date2num([start_time + timedelta(seconds=t) for t in times])
            extent = [dt_nums[0], dt_nums[-1], freqs[0], freqs[-1]]
            img = self.ax.imshow(
                Sxx_norm,
                aspect="auto",
                origin="lower",
                cmap="magma",
                extent=extent
            )
            self.ax.xaxis_date()
            self.ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
            self.ax.set_xlabel("Time (HH:MM:SS)")
        else:
            extent = [times[0], times[-1], freqs[0], freqs[-1]]
            img = self.ax.imshow(
                Sxx_norm,
                aspect="auto",
                origin="lower",
                cmap="magma",
                extent=extent
            )
            self.ax.set_xlabel("Time (s)")

        self.ax.set_ylabel("Frequency (Hz)")
        self.ax.set_title("Spectrogram")
        self.image = img

        # Re-create the info_text on a cleared axis
        self.info_text = self.ax.text(
            0.98, 0.98, "", transform=self.ax.transAxes,
            ha='right', va='top',
            color='white', fontsize=8,
            backgroundcolor='black', alpha=0.6
        )

        self.draw()

    def plot_tracks(self, freqs, times, tracks, start_time=None, color="cyan", linewidth=1.0):
        """
        Overlay each detected track on top of the spectrogram.
        If start_time is provided, convert times to matplotlib date numbers.
        """
        # Remove any previously drawn track lines
        for ln in self.track_lines:
            ln.remove()
        self.track_lines = []

        if start_time is not None:
            dt_nums = date2num([start_time + timedelta(seconds=t) for t in times])

        for track in tracks:
            time_indices = np.array([pt[0] for pt in track])
            freq_indices = [pt[1] for pt in track]
            freq_values = [freqs[idx] for idx in freq_indices]

            if start_time is not None:
                times_vals = dt_nums[time_indices]
            else:
                times_vals = time_indices * (times[1] - times[0]) + times[0]

            line, = self.ax.plot(
                times_vals,
                freq_values,
                color=color,
                linewidth=linewidth,
                picker=5  # allow picking lines for deletion
            )
            self.track_lines.append(line)

        self.draw()

    def clear_tracks(self):
        """
        Remove all tracks from the canvas.
        """
        for ln in self.track_lines:
            ln.remove()
        self.track_lines = []
        self.draw()

    def on_scroll(self, event):
        """
        Zoom In/Out based on mouse wheel.
        If Ctrl is held, zoom frequency (Y-axis). Otherwise, zoom time (X-axis).
        """
        if event.xdata is None or event.ydata is None:
            return

        base_scale = 1.1
        ctrl_held = False
        if hasattr(event, 'key') and event.key is not None:
            ctrl_held = ('control' in event.key)

        ax = self.ax
        if ctrl_held:
            # Zoom Y-axis (frequency)
            ymin, ymax = ax.get_ylim()
            if event.button == 'up':
                scale_factor = 1 / base_scale
            else:
                scale_factor = base_scale

            ydata = event.ydata
            new_ymin = ydata - (ydata - ymin) * scale_factor
            new_ymax = ydata + (ymax - ydata) * scale_factor
            ax.set_ylim(new_ymin, new_ymax)
        else:
            # Zoom X-axis (time)
            xmin, xmax = ax.get_xlim()
            if event.button == 'up':
                scale_factor = 1 / base_scale
            else:
                scale_factor = base_scale

            xdata = event.xdata
            new_xmin = xdata - (xdata - xmin) * scale_factor
            new_xmax = xdata + (xmax - xdata) * scale_factor
            ax.set_xlim(new_xmin, new_xmax)

        # After zooming, force redraw so ticks update
        self.draw()

    def on_mouse_move(self, event):
        """
        When mouse moves over the spectrogram, update the info_text
        showing Time, Frequency, and Intensity at the cursor.
        """
        if event.xdata is None or event.ydata is None:
            # Hide text if outside axes
            self.info_text.set_text("")
            self.draw()
            return

        if self.freqs is None or self.times is None or self.Sxx_norm is None:
            return

        # Determine relative seconds (rel_sec) and display Time label
        if self.start_time is not None:
            # event.xdata is matplotlib date num
            dt = num2date(event.xdata)
            # Convert to naive datetime to subtract self.start_time
            dt_naive = dt.replace(tzinfo=None)
            rel_sec = (dt_naive - self.start_time).total_seconds()
            # Format absolute time string HH:MM:SS
            time_str = dt_naive.strftime("%H:%M:%S")
        else:
            rel_sec = event.xdata
            time_str = f"{rel_sec:.2f}s"

        # Find nearest time index
        ti = np.abs(self.times - rel_sec).argmin()
        # Determine frequency value under cursor
        y_val = event.ydata
        fi = np.abs(self.freqs - y_val).argmin()
        freq_str = f"{self.freqs[fi]:.1f} Hz"

        # Intensity (normalized) at that (ti, fi)
        intensity = self.Sxx_norm[fi, ti]
        intensity_str = f"{intensity:.3f}"

        # Update info_text content
        info = f"Time: {time_str}\nFreq: {freq_str}\nIntensity: {intensity_str}"
        self.info_text.set_text(info)
        self.draw()
