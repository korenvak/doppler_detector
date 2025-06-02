# File: gui/power_spectrum_widget.py

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget, QVBoxLayout


class PowerSpectrumWidget(QWidget):
    """
    Displays a 1D power spectrum (power vs. frequency) for a single time index.
    Whenever the user selects a time slice on the spectrogram, this widget updates
    to plot Sxx_norm[:, selected_time_index].
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Power Spectrum")

        # Create a Matplotlib figure & axes
        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        self.canvas = FigureCanvas(self.fig)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

        # Internal storage for data
        self.freqs = None
        self.power = None

    def update_spectrum(self, freqs: np.ndarray, Sxx_norm: np.ndarray, time_index: int):
        """
        Given the full spectrogram Sxx_norm and its freq vector,
        plot the 1D slice at the specified time_index.
        """
        if freqs is None or Sxx_norm is None or time_index < 0 or time_index >= Sxx_norm.shape[1]:
            return

        self.freqs = freqs
        self.power = Sxx_norm[:, time_index]

        self.ax.clear()
        self.ax.plot(self.freqs, self.power)
        self.ax.set_title(f"Power Spectrum (t_idx = {time_index})")
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Normalized Power")
        self.ax.grid(True)

        self.canvas.draw()
