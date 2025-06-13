class DetectionManager:
    """Store detection results for undo operations."""

    def __init__(self):
        self.history = []  # each entry is a list of PlotDataItems

    def record(self, items):
        self.history.append(items)

    def undo_last(self, canvas):
        if not self.history:
            return
        items = self.history.pop()
        for it in items:
            try:
                canvas.plot.removeItem(it)
            except Exception:
                pass
