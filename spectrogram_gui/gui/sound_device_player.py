import numpy as np
import sounddevice as sd
import threading
import time
import os

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSlider, QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from doppler_detector.spectrogram_gui.utils.ffmpeg_utils import convert_to_wav
import soundfile as sf


class SoundDevicePlayer(QWidget):
    """
    Audio playback widget using sounddevice.
    Exposes methods to retrieve and replace the waveform for filtering/FFT/gain.
    """

    prevRequested = pyqtSignal()
    nextRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None            # numpy array of audio samples
        self.sample_rate = None     # sample rate int
        self.channels = 1
        self.stream = None
        self.position = 0           # in milliseconds
        self.start_time = 0         # timestamp when playback started
        self.lock = threading.Lock()
        self.playing = False
        self.position_callback = None
        self.duration_ms = 0
        self._user_dragging = False
        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(50)
        self._ui_timer.timeout.connect(self._on_ui_tick)

        # Layout + playback/navigation buttons
        self.layout = QHBoxLayout(self)
        self.layout.setSpacing(12)
        self.layout.setAlignment(Qt.AlignCenter)
        # Seek slider + time label
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderPressed.connect(self._on_slider_pressed)
        self.position_slider.sliderReleased.connect(self._on_slider_released)
        self.position_slider.sliderMoved.connect(self._on_slider_moved)
        self.time_label = QLabel("00:00 / 00:00")

        self.prev_btn = QPushButton("⏮")
        self.next_btn = QPushButton("⏭")
        self.play_btn = QPushButton("▶ Play")
        self.stop_btn = QPushButton("⏹ Stop")
        self.play_btn.clicked.connect(self.play)
        self.stop_btn.clicked.connect(self.stop)
        # Emit navigation signals on button clicks
        self.prev_btn.clicked.connect(lambda: self.prevRequested.emit())
        self.next_btn.clicked.connect(lambda: self.nextRequested.emit())
        self.layout.addWidget(self.prev_btn)
        self.layout.addWidget(self.play_btn)
        self.layout.addWidget(self.stop_btn)
        self.layout.addWidget(self.next_btn)
        self.layout.addWidget(self.position_slider, 1)
        self.layout.addWidget(self.time_label)

    def load(self, filepath):
        """
        Convert to WAV (via ffmpeg_utils), then read into numpy+sf.
        """
        print(f"[SoundDevicePlayer] Loading file: {filepath}")
        wav_path = convert_to_wav(filepath)
        if not wav_path or not os.path.exists(wav_path):
            print("[SoundDevicePlayer] Failed to convert file.")
            return

        try:
            self.data, self.sample_rate = sf.read(wav_path, dtype='float32')
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

        self.channels = 1 if self.data.ndim == 1 else self.data.shape[1]
        self.position = 0
        self.start_time = time.time()
        self.duration_ms = int(1000 * (self.data.shape[0] if self.channels == 1 else self.data.shape[0]) / self.sample_rate)
        self.position_slider.setRange(0, self.duration_ms)
        self._update_time_label()
        print(
            f"[SoundDevicePlayer] Loaded: {filepath}, shape={self.data.shape}, sr={self.sample_rate}"
        )

    def play(self):
        """
        Start playback from self.position (in ms).
        """
        if self.stream is not None:
            self.stop()
        if self.data is None or self.sample_rate is None:
            return

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._callback
        )
        self.start_time = time.time() - (self.position / 1000.0)
        self.playing = True
        self.stream.start()
        self._ui_timer.start()

    def stop(self):
        """
        Stop playback immediately.
        """
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.playing = False
        self._ui_timer.stop()

    def seek(self, ms):
        """
        Seek playback to a given time (milliseconds).
        """
        with self.lock:
            self.position = ms
            self.start_time = time.time() - (ms / 1000.0)
            if self.playing:
                self.stop()
                self.play()

    def _on_ui_tick(self):
        """Update slider and label on the UI thread."""
        if not self._user_dragging:
            self.position_slider.setValue(int(self.position))
        self._update_time_label()

    def _update_time_label(self):
        def fmt(ms):
            sec = max(0, int(ms // 1000))
            return f"{sec//60:02d}:{sec%60:02d}"
        self.time_label.setText(f"{fmt(self.position)} / {fmt(self.duration_ms)}")

    def _on_slider_pressed(self):
        self._user_dragging = True

    def _on_slider_released(self):
        self._user_dragging = False
        target = self.position_slider.value()
        self.seek(int(target))

    def _on_slider_moved(self, value):
        # update preview label while dragging
        self.position = int(value)
        self._update_time_label()

    def _callback(self, outdata, frames, time_info, status):
        """
        sounddevice callback: fill outdata with data from self.data,
        padded with zeros if we hit the end.
        """
        with self.lock:
            if self.data is None:
                outdata.fill(0)
                return

            start_sample = int(self.position * self.sample_rate / 1000)
            end_sample = start_sample + frames
            chunk = self.data[start_sample:end_sample]

            if len(chunk) < frames:
                chunk = np.pad(
                    chunk,
                    ((0, frames - len(chunk)),) if self.channels == 1 else ((0, frames - len(chunk)), (0, 0)),
                    mode='constant'
                )
                self.playing = False
                raise sd.CallbackStop()

            if self.channels == 1:
                chunk = chunk.reshape(-1, 1)

            outdata[:] = chunk
            self.position += int(1000 * frames / self.sample_rate)

            if self.position_callback:
                self.position_callback(self.position)

    def set_position_callback(self, callback):
        """
        Provide a function(callback_ms) that will be called each buffer,
        so canvas can update its playback cursor.
        """
        self.position_callback = callback

    def get_waveform_copy(self, return_sr=False):
        """
        Return a copy of the current waveform (and sample rate if requested).
        """
        if self.data is None:
            return (None, None) if return_sr else None
        if return_sr:
            return self.data.copy(), self.sample_rate
        return self.data.copy()

    def replace_waveform(self, new_wave):
        """
        Replace the internal waveform with new_wave (same sample_rate).
        Restart playback from beginning if currently playing.
        """
        self.data = new_wave.astype(np.float32)
        self.position = 0
        if self.playing:
            self.stop()
            self.play()
