import os
from datetime import datetime
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPixmap

def save_snapshot(canvas,
                  start_dt: datetime,
                  end_dt: datetime,
                  pixel: int,
                  index: int,
                  snapshot_path: str = None) -> str:
    """
    Take a screenshot of the *entire* spectrogram canvas widget exactly
    as you see it (including current time/freq zoom, annotation lines, filters)
    and save it to `snapshot_path` (or default folder). Returns the path.
    """
    # 1) Build default path if none given
    if snapshot_path is None:
        folder = os.path.join(os.path.expanduser("~"), "spectrogram_snapshots")
        os.makedirs(folder, exist_ok=True)
        tstr  = start_dt.strftime("%H-%M-%S")
        mms   = f"{start_dt.microsecond // 100:03d}"
        fname = f"{pixel}_{tstr}_{mms}.png"
        snapshot_path = os.path.join(folder, fname)
    else:
        os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)

    # 2) Let Qt finish any pending redraws (so your annotations are on screen)
    QApplication.processEvents()

    # 3) Grab the widget itself (this captures exactly what you see)
    #    We assume `canvas` is a QWidget (e.g. a GraphicsLayoutWidget or
    #    PlotWidget). If not, fall back to grabbing the ViewBox viewport.
    if hasattr(canvas, 'grab'):
        pixmap: QPixmap = canvas.grab()
    else:
        views = canvas.plot.scene().views()
        if not views:
            raise RuntimeError("Cannot snapshot: no QGraphicsView found in scene()")
        view = views[0]                       # QGraphicsView
        pixmap: QPixmap = view.viewport().grab()

    # 4) Save to disk
    pixmap.save(snapshot_path)
    print(f"[Snapshot] Saved screenshot to: {snapshot_path}")

    return snapshot_path
