from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton
import numpy as np
import pyqtgraph as pg


class FFTDialog(QDialog):
    """
    Dialog that computes and displays the FFT of the selected segment.
    """

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.setWindowTitle("FFT of Selected Segment")
        self.resize(600, 400)

        layout = QVBoxLayout(self)

        # Close button
        self.close_btn = QPushButton("Close")
        layout.addWidget(self.close_btn)
        self.close_btn.clicked.connect(self.accept)

        # Immediately compute & show FFT
        self.plot_fft()

    def plot_fft(self):
        # Selected time range
        t0, t1 = self.main_window.canvas.selected_range

        # Current waveform + sr
        wave = self.main_window.audio_player.data
        sr = self.main_window.audio_player.sample_rate

        total_duration = len(wave) / sr
        idx0 = int(np.floor((t0 / total_duration) * len(wave)))
        idx1 = int(np.ceil((t1 / total_duration) * len(wave)))
        idx0 = max(0, min(idx0, len(wave) - 1))
        idx1 = max(0, min(idx1, len(wave)))

        segment = wave[idx0:idx1]
        if len(segment) == 0:
            return

        # Apply Hann window, compute FFT
        N = len(segment)
        windowed = segment * np.hanning(N)
        yf = np.fft.rfft(windowed)
        xf = np.fft.rfftfreq(N, 1.0 / sr)
        magnitude = np.abs(yf) / N

        # Display using pyqtgraph
        plot_widget = pg.PlotWidget(title="FFT Magnitude (Log scale)")
        plot_widget.plot(xf, magnitude, pen='c')
        plot_widget.setLogMode(x=False, y=True)
        plot_widget.setLabel('bottom', 'Frequency', units='Hz')
        plot_widget.setLabel('left', 'Magnitude')
        self.layout().addWidget(plot_widget)
