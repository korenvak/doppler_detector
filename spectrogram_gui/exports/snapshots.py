from pathlib import Path
from spectrogram_gui.utils.snapshot_utils import save_snapshot

def save_canvas_snapshot(canvas, start_dt, end_dt, pixel, index, out_path=None):
    path = save_snapshot(canvas, start_dt, end_dt, pixel, index, snapshot_path=out_path)
    return path
