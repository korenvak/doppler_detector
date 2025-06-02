import numpy as np
from PyQt5.QtWidgets import QMessageBox

class EventEditor:
    def __init__(self, canvas, freqs, times, start_time=None):
        """
        Manages interactive editing of detected tracks.
        Supports:
          1. 'Delete Mode' (Edit Mode) – delete on click
          2. 'Add Mode' – user מוסיף נקודות מסלול ידני
          3. Undo last edit
        """
        self.canvas = canvas
        self.freqs = freqs
        self.times = times
        self.start_time = start_time

        # Current list of tracks
        self.tracks = []

        # History stack for Undo (list of track-lists)
        self.history = []

        # Modes
        self.delete_mode = False
        self.add_mode = False
        self.new_track_points = []  # Holds [(ti, fi), …] while building a manual track

        # Connect pick_event for deletion
        self.canvas.mpl_connect("pick_event", self.on_pick)
        # Connect mouse clicks for manual add
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)

    def set_tracks(self, tracks):
        """
        Set the current list of tracks and draw them on the canvas.
        Also clear history and push initial state.
        """
        self.tracks = [t.copy() for t in tracks]
        self.history = [ [t.copy() for t in tracks] ]
        self.canvas.plot_tracks(self.freqs, self.times, self.tracks, start_time=self.start_time)

    def on_pick(self, event):
        """
        Delete the picked track only if delete_mode is True.
        """
        if not self.delete_mode:
            return

        picked_line = event.artist
        idx_to_remove = None
        for idx, line in enumerate(self.canvas.track_lines):
            if line == picked_line:
                idx_to_remove = idx
                break

        if idx_to_remove is not None:
            # Push current state to history (for Undo)
            self.history.append([t.copy() for t in self.tracks])

            del self.tracks[idx_to_remove]
            self.canvas.plot_tracks(self.freqs, self.times, self.tracks, start_time=self.start_time)
            QMessageBox.information(
                self.canvas,
                "Event Removed",
                f"Removed event #{idx_to_remove + 1}"
            )

    def on_canvas_click(self, event):
        """
        In Add Mode: user יוצר מסלול ידני.
        כל לחיצה מוסיפה נקודה אחת (time_idx, freq_idx) למערך new_track_points.
        כשיוצרים track מלא (לחיצה עם Shift מסמנת סיום), מוסיפים אותו לרשימת ה־tracks.
        """
        if not self.add_mode:
            return

        if event.xdata is None or event.ydata is None:
            return

        # Determine relative seconds from xdata
        if self.start_time is not None:
            # Convert event.xdata (matplotlib date num) to datetime, then to relative seconds
            from matplotlib.dates import num2date
            current_dt = num2date(event.xdata)
            rel_seconds = (current_dt - self.start_time).total_seconds()
        else:
            rel_seconds = event.xdata

        # Find nearest ti
        dt_frame = self.times[1] - self.times[0]
        ti = int(round(rel_seconds / dt_frame))
        ti = max(0, min(len(self.times) - 1, ti))

        # Find nearest fi by comparing event.ydata to self.freqs
        freq_clicked = event.ydata
        fi = np.abs(self.freqs - freq_clicked).argmin()

        # Append to new_track_points
        self.new_track_points.append((ti, fi))

        # If user holds Shift while clicking – we finalize the new track
        if event.key == "shift":
            if len(self.new_track_points) >= 2:
                # Sort points by time
                new_track = sorted(self.new_track_points, key=lambda x: x[0])
                # Push history
                self.history.append([t.copy() for t in self.tracks])
                # Add to tracks
                self.tracks.append(new_track)
                # Redraw
                self.canvas.plot_tracks(self.freqs, self.times, self.tracks, start_time=self.start_time)
                QMessageBox.information(
                    self.canvas,
                    "Event Added",
                    f"Added manual event with {len(new_track)} points"
                )
            # Reset new_track_points whether or not it was valid
            self.new_track_points = []

    def undo(self):
        """
        Undo last edit (delete or add). Pop from history.
        """
        if len(self.history) < 2:
            return  # Nothing to undo

        # Pop current state
        self.history.pop()
        # Restore previous
        prev = [t.copy() for t in self.history[-1]]
        self.tracks = prev
        self.canvas.plot_tracks(self.freqs, self.times, self.tracks, start_time=self.start_time)

    def set_delete_mode(self, enabled: bool):
        """
        Turn Delete Mode on/off.
        """
        self.delete_mode = enabled
        if enabled:
            self.add_mode = False  # cannot be both

    def set_add_mode(self, enabled: bool):
        """
        Turn Add Mode on/off.
        """
        self.add_mode = enabled
        if enabled:
            self.delete_mode = False
            self.new_track_points = []
