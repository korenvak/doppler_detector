import numpy as np
import sounddevice as sd
import threading
import time
import os

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSlider, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from spectrogram_gui.utils.ffmpeg_utils import convert_to_wav
import soundfile as sf


class SoundDevicePlayer(QWidget):
    """
    Enhanced audio playback widget using sounddevice.
    Features:
    - Play/Pause/Stop controls
    - Seek slider with time display
    - Volume control
    - Previous/Next navigation
    - Exposes methods to retrieve and replace the waveform for filtering/FFT/gain.
    """

    prevRequested = pyqtSignal()
    nextRequested = pyqtSignal()
    positionChanged = pyqtSignal(float)  # Signal for position updates (in seconds)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.data = None            # numpy array of audio samples
        self.sample_rate = None     # sample rate int
        self.channels = 1
        self.stream = None
        self.position = 0           # in milliseconds
        self.duration = 0           # total duration in milliseconds
        self.start_time = 0         # timestamp when playback started
        self.lock = threading.Lock()
        self.playing = False
        self.paused = False
        self.position_callback = None
        self.volume = 1.0           # Volume level (0.0 to 1.0)
        
        # Timer for updating position
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_position)
        self.update_timer.setInterval(100)  # Update every 100ms

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Control buttons layout
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)
        controls_layout.setAlignment(Qt.AlignCenter)
        
        # Navigation and playback buttons
        self.prev_btn = QPushButton("⏮")
        self.prev_btn.setToolTip("Previous File")
        self.prev_btn.setFixedWidth(40)
        
        self.play_pause_btn = QPushButton("▶")
        self.play_pause_btn.setToolTip("Play/Pause")
        self.play_pause_btn.setFixedWidth(50)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        
        self.stop_btn = QPushButton("⏹")
        self.stop_btn.setToolTip("Stop")
        self.stop_btn.setFixedWidth(40)
        self.stop_btn.clicked.connect(self.stop)
        
        self.next_btn = QPushButton("⏭")
        self.next_btn.setToolTip("Next File")
        self.next_btn.setFixedWidth(40)
        
        self.prev_btn.clicked.connect(self.prevRequested)
        self.next_btn.clicked.connect(self.nextRequested)
        
        # Time labels
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setMinimumWidth(100)
        self.time_label.setAlignment(Qt.AlignCenter)
        
        # Position slider
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 1000)
        self.position_slider.sliderPressed.connect(self._on_slider_pressed)
        self.position_slider.sliderReleased.connect(self._on_slider_released)
        self.position_slider.valueChanged.connect(self._on_slider_moved)
        self.slider_being_dragged = False
        
        # Volume control
        self.volume_label = QLabel("🔊")
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setFixedWidth(80)
        self.volume_slider.setToolTip("Volume")
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        
        # Add widgets to controls layout
        controls_layout.addWidget(self.prev_btn)
        controls_layout.addWidget(self.play_pause_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.next_btn)
        controls_layout.addWidget(self.time_label)
        controls_layout.addWidget(self.position_slider, 1)  # Stretch the slider
        controls_layout.addWidget(self.volume_label)
        controls_layout.addWidget(self.volume_slider)
        
        main_layout.addLayout(controls_layout)

    def load(self, filepath):
        """
        Convert to WAV (via ffmpeg_utils), then read into numpy+sf.
        """
        print(f"[SoundDevicePlayer] Loading file: {filepath}")
        
        # Stop any current playback
        self.stop()
        
        wav_path = convert_to_wav(filepath)
        if not wav_path or not os.path.exists(wav_path):
            print("[SoundDevicePlayer] Failed to convert file.")
            return False

        try:
            self.data, self.sample_rate = sf.read(wav_path, dtype='float32')
        except Exception as e:
            print(f"[SoundDevicePlayer] Error reading file: {e}")
            return False
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

        self.channels = 1 if self.data.ndim == 1 else self.data.shape[1]
        self.position = 0
        self.duration = int(len(self.data) * 1000 / self.sample_rate)
        self.start_time = time.time()
        
        # Update UI
        self.position_slider.setMaximum(self.duration)
        self.position_slider.setValue(0)
        self._update_time_label()
        self.play_pause_btn.setText("▶")
        self.play_pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.position_slider.setEnabled(True)
        
        print(
            f"[SoundDevicePlayer] Loaded: {filepath}, shape={self.data.shape}, sr={self.sample_rate}, duration={self.duration/1000:.2f}s"
        )
        return True

    def play(self):
        """
        Start playback from self.position (in ms).
        """
        if self.data is None or self.sample_rate is None:
            return
        
        if self.stream is not None and self.paused:
            # Resume from pause
            self.paused = False
            self.playing = True
            self.start_time = time.time() - (self.position / 1000.0)
            self.stream.start()
        else:
            # Start new playback
            if self.stream is not None:
                self.stop()
            
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self._callback
            )
            self.start_time = time.time() - (self.position / 1000.0)
            self.playing = True
            self.paused = False
            self.stream.start()
        
        self.play_pause_btn.setText("⏸")
        self.update_timer.start()
    
    def pause(self):
        """
        Pause playback.
        """
        if self.stream is not None and self.playing:
            self.stream.stop()
            self.playing = False
            self.paused = True
            self.play_pause_btn.setText("▶")
            self.update_timer.stop()
    
    def toggle_play_pause(self):
        """
        Toggle between play and pause.
        """
        if self.playing:
            self.pause()
        else:
            self.play()

    def stop(self):
        """
        Stop playback immediately and reset position.
        """
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        self.playing = False
        self.paused = False
        self.position = 0
        self.play_pause_btn.setText("▶")
        self.update_timer.stop()
        
        if not self.slider_being_dragged:
            self.position_slider.setValue(0)
        self._update_time_label()

    def seek(self, ms):
        """
        Seek playback to a given time (milliseconds).
        """
        with self.lock:
            self.position = max(0, min(ms, self.duration))
            self.start_time = time.time() - (self.position / 1000.0)
            
            if not self.slider_being_dragged:
                self.position_slider.setValue(self.position)
            self._update_time_label()
            
            # Emit position change signal
            if self.position_callback:
                self.position_callback(self.position)
            self.positionChanged.emit(self.position / 1000.0)

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
                # Reached end of audio
                chunk = np.pad(
                    chunk,
                    ((0, frames - len(chunk)),) if self.channels == 1 else ((0, frames - len(chunk)), (0, 0)),
                    mode='constant'
                )
                self.playing = False
                self.position = self.duration
                raise sd.CallbackStop()

            if self.channels == 1:
                chunk = chunk.reshape(-1, 1)

            # Apply volume
            chunk = chunk * self.volume
            
            outdata[:] = chunk
            self.position = min(self.position + int(1000 * frames / self.sample_rate), self.duration)

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
        was_playing = self.playing
        self.stop()
        
        self.data = new_wave.astype(np.float32)
        self.channels = 1 if self.data.ndim == 1 else self.data.shape[1]
        self.position = 0
        self.duration = int(len(self.data) * 1000 / self.sample_rate) if self.sample_rate else 0
        
        # Update UI
        self.position_slider.setMaximum(self.duration)
        self.position_slider.setValue(0)
        self._update_time_label()
        
        if was_playing:
            self.play()
    
    def _update_position(self):
        """
        Update position display during playback.
        """
        if self.playing and not self.slider_being_dragged:
            elapsed = (time.time() - self.start_time) * 1000
            self.position = min(int(elapsed), self.duration)
            self.position_slider.setValue(self.position)
            self._update_time_label()
            
            if self.position_callback:
                self.position_callback(self.position)
            self.positionChanged.emit(self.position / 1000.0)
            
            if self.position >= self.duration:
                self.stop()
    
    def _update_time_label(self):
        """
        Update the time display label.
        """
        current = self.position / 1000.0
        total = self.duration / 1000.0
        
        current_str = f"{int(current // 60):02d}:{int(current % 60):02d}"
        total_str = f"{int(total // 60):02d}:{int(total % 60):02d}"
        
        self.time_label.setText(f"{current_str} / {total_str}")
    
    def _on_slider_pressed(self):
        """
        Called when user starts dragging the slider.
        """
        self.slider_being_dragged = True
    
    def _on_slider_released(self):
        """
        Called when user releases the slider.
        """
        self.slider_being_dragged = False
        self.seek(self.position_slider.value())
    
    def _on_slider_moved(self, value):
        """
        Called when slider value changes.
        """
        if self.slider_being_dragged:
            self.position = value
            self._update_time_label()
    
    def _on_volume_changed(self, value):
        """
        Called when volume slider changes.
        """
        self.volume = value / 100.0
