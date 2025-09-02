import os
import pandas as pd
from PyQt5.QtWidgets import QInputDialog, QLineEdit, QDialog
from PyQt5.QtCore import Qt, QTimer
from datetime import timedelta
import pyqtgraph as pg

from spectrogram_gui.utils.snapshot_utils import save_snapshot
from spectrogram_gui.gui.annotation_editor import AnnotationEditorDialog

class EventAnnotator:
    def __init__(self, canvas, undo_callback=None):
        """
        Initialize with a reference to the SpectrogramCanvas.
        """
        self.canvas = canvas
        self.csv_path = None
        # now includes Snapshot column
        self.df = pd.DataFrame(
            columns=["Start", "End", "Site", "Pixel", "Type", "Description", "Snapshot"]
        )
        self.start_time_pending = None
        self.metadata = {}
        self.undo_callback = undo_callback
        self.graphics_history = []  # keep track of plot items per annotation
        self.current_items = []

    def set_metadata(self, site, pixel, file_start):
        """
        Provide site, pixel, and file start datetime for upcoming annotations.
        """
        self.metadata["site"] = site
        self.metadata["pixel"] = pixel
        self.metadata["file_start"] = file_start

    def set_csv_path(self, path):
        """
        Set the CSV output path and try to load existing annotations if the file exists.
        """
        self.csv_path = path
        if os.path.exists(path):
            try:
                self.df = pd.read_csv(path)
            except Exception:
                pass

    def on_click(self, rel_sec, event):
        """
        Handle click events to create start/end annotation markers.
        First click: record start time (relative seconds) and draw a dashed line.
        Second click: record end time, prompt for type and description,
        add to DataFrame (including snapshot path), show the Annotation Editor dialog,
        save CSV and schedule snapshot saving.
        """
        file_start = self.metadata.get("file_start")
        if file_start is None or self.canvas.times is None:
            print("[Mark Event] Cannot mark — no file or canvas time initialized.")
            return

        # compute absolute x
        try:
            x_pos = self.canvas.times[0] + rel_sec
            if not (0 < x_pos < 1e10):
                raise ValueError("x out of bounds")
        except Exception as e:
            print(f"[Mark Event] Invalid x for drawing: {e}")
            return

        # First click → start
        if self.start_time_pending is None:
            self.start_time_pending = rel_sec
            y = self.canvas.freqs[-1] * 0.95 if self.canvas.freqs is not None else 0

            label = pg.TextItem(
                f"Annotation {len(self.df) + 1} start",
                anchor=(0, 0),
                fill=pg.mkBrush(0, 0, 0, 200)
            )
            label.setPos(x_pos, y)
            self.canvas.plot.addItem(label)

            line = pg.InfiniteLine(
                pos=x_pos,
                angle=90,
                pen=pg.mkPen(color=(0, 255, 255), style=Qt.DashLine)
            )
            self.canvas.plot.addItem(line)
            self.current_items = [label, line]
            return

        # Second click → end
        start_rel = self.start_time_pending
        end_rel = rel_sec
        self.start_time_pending = None

        if end_rel < start_rel:
            start_rel, end_rel = end_rel, start_rel

        start_dt = file_start + timedelta(seconds=start_rel)
        end_dt   = file_start + timedelta(seconds=end_rel)

        # ask type & description
        ev_type, ok1 = QInputDialog.getText(
            None, "Event Type", "Enter type:", QLineEdit.Normal
        )
        if not ok1 or not ev_type.strip():
            return

        desc, ok2 = QInputDialog.getText(
            None, "Description", "Enter description:", QLineEdit.Normal
        )
        if not ok2:
            return

        # build snapshot filename (cross-platform, under user's home by default)
        timestr = start_dt.strftime("%H-%M-%S_%f")[:-3]  # HH-MM-SS_mmm
        base_dir = os.path.join(os.path.expanduser("~"), "spectrogram_snapshots")
        os.makedirs(base_dir, exist_ok=True)
        fname = f"{self.metadata['pixel']}_{timestr}.png"
        snapshot_path = os.path.join(base_dir, fname)

        # build row (using local-naive datetime format without microseconds)
        row = {
            "Start":    start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "End":      end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Site":     self.metadata["site"],
            "Pixel":    self.metadata["pixel"],
            "Type":     ev_type,
            "Description": desc,
            "Snapshot": snapshot_path
        }

        # Append to DataFrame
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

        # save CSV & open editor
        if self.csv_path:
            try:
                os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
                self.df.to_csv(self.csv_path, index=False)
            except Exception as e:
                print(f"[CSV Error] {e}")

            dialog = AnnotationEditorDialog(self.df, parent=None)
            if dialog.exec_() == QDialog.Accepted:
                try:
                    self.df.to_csv(self.csv_path, index=False)
                except Exception as e:
                    print(f"[CSV Error] {e}")

        # schedule snapshot export (with annotations visible)
        def save_later():
            try:
                print(f"[Snapshot] Saving snapshot from {start_dt} to {end_dt}")
                save_snapshot(
                    self.canvas,
                    start_dt,
                    end_dt,
                    self.metadata["pixel"],
                    len(self.df),
                    snapshot_path=snapshot_path
                )
            except Exception as e:
                print(f"[Snapshot Error] {e}")

        QTimer.singleShot(100, save_later)

        # draw end‐label & end‐line
        x_end = self.canvas.times[0] + end_rel
        y = self.canvas.freqs[-1] * 0.95 if self.canvas.freqs is not None else 0

        label = pg.TextItem(
            f"Annotation {len(self.df)} end",
            anchor=(0, 0),
            fill=pg.mkBrush(0, 0, 0, 200)
        )
        label.setPos(x_end, y)
        self.canvas.plot.addItem(label)

        line = pg.InfiniteLine(
            pos=x_end,
            angle=90,
            pen=pg.mkPen(color=(0, 255, 255), style=Qt.DashLine)
        )
        self.canvas.plot.addItem(line)

        items = self.current_items + [label, line]
        self.graphics_history.append(items)
        self.current_items = []
        if self.undo_callback:
            self.undo_callback(("annotation", None))

    def count(self):
        """Return the number of annotations recorded so far."""
        return len(self.df)

    def undo_last(self):
        if not self.graphics_history:
            return
        items = self.graphics_history.pop()
        for it in items:
            try:
                self.canvas.plot.removeItem(it)
            except Exception:
                pass
        if not self.df.empty:
            self.df = self.df.iloc[:-1].copy()
            if self.csv_path:
                try:
                    self.df.to_csv(self.csv_path, index=False)
                except Exception:
                    pass
