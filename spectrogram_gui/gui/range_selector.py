# File: range_selector.py

from PySide6.QtCore import QObject, Sig, Signalnal, Qt
import pyqtgraph as pg


class RangeSelector(QObject):
    """
    A helper to let the user drag a vertical region over a pyqtgraph PlotItem
    and emit the corresponding (t_start, t_end) in data coordinates.
    """
    range_changed = Signal(float, float)

    def __init__(self, plot_item: pg.PlotItem):
        super().__init__()
        self.plot_item = plot_item

        # Create a LinearRegionItem (vertical region) for selecting time range
        self.lr = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Vertical, brush=(50, 50, 200, 50))
        self.lr.setZValue(10)
        self.lr.hide()
        self.plot_item.addItem(self.lr)

        # Connect signal when region changes
        self.lr.sigRegionChanged.connect(self.on_region_changed)

        # Enable right-click to create region
        self.plot_item.scene().sigMouseClicked.connect(self.on_mouse_click)

    def on_mouse_click(self, event):
        """
        On right-click, initialize a small region at the clicked x-coordinate,
        then allow the user to resize/drag.
        """
        if event.button() == Qt.RightButton:
            pos = event.scenePos()
            mouse_point = self.plot_item.vb.mapSceneToView(pos)
            x = mouse_point.x()

            view_range = self.plot_item.viewRange()[0]
            left_limit, right_limit = view_range[0], view_range[1]
            width = min(5, (right_limit - left_limit) / 4)  # default 0.5s or smaller
            low = max(left_limit, x - width / 2)
            high = min(right_limit, x + width / 2)
            self.lr.setRegion([low, high])
            self.lr.show()
            self.on_region_changed()

    def on_region_changed(self):
        """
        Called when the user drags/resizes the LinearRegionItem.
        Convert region to data coordinates and emit.
        """
        reg = self.lr.getRegion()  # [minX, maxX]
        t_start, t_end = reg[0], reg[1]
        self.range_changed.emit(t_start, t_end)

    def clear(self):
        """
        Hide the region selector.
        """
        self.lr.hide()
