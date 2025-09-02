"""
Modern audio player with sounddevice integration
Provides smooth playback with position tracking and controls
"""

import sounddevice as sd
import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer, QThread, QMutex, QWaitCondition
import threading
import queue


class AudioPlaybackThread(QThread):
    """
    Dedicated thread for audio playback to prevent UI blocking
    """
    position_updated = Signal(float)  # Current position in seconds
    playback_finished = Signal()
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.audio_data = None
        self.sample_rate = 44100
        self.is_playing = False
        self.is_paused = False
        self.current_position = 0
        self.loop = False
        self.volume = 1.0
        
        # Thread synchronization
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        self.stop_flag = False
        
        # Audio stream
        self.stream = None
        self.position_lock = threading.Lock()
        
    def set_audio(self, audio_data, sample_rate):
        """Set audio data for playback"""
        with self.position_lock:
            self.audio_data = audio_data.astype(np.float32)
            self.sample_rate = sample_rate
            self.current_position = 0
            
    def set_volume(self, volume):
        """Set playback volume (0.0 to 1.0)"""
        self.volume = np.clip(volume, 0.0, 1.0)
        
    def seek(self, position_seconds):
        """Seek to specific position in seconds"""
        with self.position_lock:
            if self.audio_data is not None:
                sample_position = int(position_seconds * self.sample_rate)
                self.current_position = np.clip(
                    sample_position, 0, len(self.audio_data)
                )
                
    def play(self):
        """Start or resume playback"""
        self.is_playing = True
        self.is_paused = False
        self.wait_condition.wakeAll()
        
    def pause(self):
        """Pause playback"""
        self.is_paused = True
        
    def stop(self):
        """Stop playback and reset position"""
        self.is_playing = False
        self.is_paused = False
        self.current_position = 0
        if self.stream:
            self.stream.stop()
            
    def run(self):
        """Main playback loop"""
        while not self.stop_flag:
            self.mutex.lock()
            
            if not self.is_playing or self.is_paused:
                self.wait_condition.wait(self.mutex)
                self.mutex.unlock()
                continue
                
            self.mutex.unlock()
            
            if self.audio_data is None:
                continue
                
            try:
                # Create audio stream
                self.stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    callback=self.audio_callback,
                    blocksize=2048,
                    finished_callback=self.stream_finished
                )
                
                with self.stream:
                    # Update position while playing
                    while self.is_playing and not self.is_paused:
                        with self.position_lock:
                            pos_seconds = self.current_position / self.sample_rate
                            self.position_updated.emit(pos_seconds)
                        self.msleep(50)  # Update every 50ms
                        
            except Exception as e:
                self.error_occurred.emit(str(e))
                self.is_playing = False
                
    def audio_callback(self, outdata, frames, time, status):
        """Audio stream callback"""
        if status:
            print(f"Audio callback status: {status}")
            
        with self.position_lock:
            if self.audio_data is None or not self.is_playing or self.is_paused:
                outdata[:] = 0
                return
                
            # Calculate how many samples we can provide
            available = len(self.audio_data) - self.current_position
            to_read = min(frames, available)
            
            if to_read > 0:
                # Copy audio data with volume applied
                audio_chunk = self.audio_data[
                    self.current_position:self.current_position + to_read
                ]
                outdata[:to_read, 0] = audio_chunk * self.volume
                
                # Zero-fill if we don't have enough samples
                if to_read < frames:
                    outdata[to_read:] = 0
                    
                self.current_position += to_read
                
                # Check if we've reached the end
                if self.current_position >= len(self.audio_data):
                    if self.loop:
                        self.current_position = 0
                    else:
                        self.is_playing = False
                        self.playback_finished.emit()
            else:
                outdata[:] = 0
                self.is_playing = False
                self.playback_finished.emit()
                
    def stream_finished(self):
        """Called when stream finishes"""
        pass
        
    def quit(self):
        """Clean shutdown"""
        self.stop_flag = True
        self.is_playing = False
        self.wait_condition.wakeAll()
        if self.stream:
            self.stream.close()
        super().quit()


class ModernAudioPlayer(QObject):
    """
    High-level audio player interface
    """
    position_changed = Signal(float, float)  # current, total
    state_changed = Signal(str)  # 'playing', 'paused', 'stopped'
    error = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Playback thread
        self.playback_thread = AudioPlaybackThread()
        self.playback_thread.position_updated.connect(self.on_position_update)
        self.playback_thread.playback_finished.connect(self.on_playback_finished)
        self.playback_thread.error_occurred.connect(self.error.emit)
        self.playback_thread.start()
        
        # State
        self.audio_data = None
        self.sample_rate = None
        self.duration = 0
        self.state = 'stopped'
        
        # Position update timer for smooth UI updates
        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self.update_position)
        self.position_timer.setInterval(100)  # Update every 100ms
        
    def load_audio(self, audio_data, sample_rate):
        """Load audio for playback"""
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.duration = len(audio_data) / sample_rate
        
        self.playback_thread.set_audio(audio_data, sample_rate)
        
    def play(self):
        """Start playback"""
        if self.audio_data is not None:
            self.playback_thread.play()
            self.state = 'playing'
            self.state_changed.emit(self.state)
            self.position_timer.start()
            
    def pause(self):
        """Pause playback"""
        self.playback_thread.pause()
        self.state = 'paused'
        self.state_changed.emit(self.state)
        self.position_timer.stop()
        
    def stop(self):
        """Stop playback"""
        self.playback_thread.stop()
        self.state = 'stopped'
        self.state_changed.emit(self.state)
        self.position_timer.stop()
        self.position_changed.emit(0, self.duration)
        
    def seek(self, position):
        """Seek to position in seconds"""
        self.playback_thread.seek(position)
        
    def set_volume(self, volume):
        """Set volume (0.0 to 1.0)"""
        self.playback_thread.set_volume(volume)
        
    def set_loop(self, loop):
        """Enable/disable loop playback"""
        self.playback_thread.loop = loop
        
    def on_position_update(self, position):
        """Handle position updates from playback thread"""
        self.position_changed.emit(position, self.duration)
        
    def on_playback_finished(self):
        """Handle playback finished"""
        self.state = 'stopped'
        self.state_changed.emit(self.state)
        self.position_timer.stop()
        self.position_changed.emit(self.duration, self.duration)
        
    def update_position(self):
        """Request position update"""
        # Position is automatically updated by the playback thread
        pass
        
    def cleanup(self):
        """Clean up resources"""
        self.position_timer.stop()
        self.playback_thread.quit()
        self.playback_thread.wait()
        
        
class AudioDeviceManager:
    """
    Manage audio devices and settings
    """
    
    @staticmethod
    def get_devices():
        """Get list of available audio devices"""
        devices = sd.query_devices()
        output_devices = []
        
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                output_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_output_channels'],
                    'sample_rate': device['default_samplerate']
                })
                
        return output_devices
        
    @staticmethod
    def set_default_device(device_id):
        """Set default output device"""
        try:
            sd.default.device = device_id
            return True
        except Exception as e:
            print(f"Failed to set audio device: {e}")
            return False
            
    @staticmethod
    def get_default_device():
        """Get current default device"""
        return sd.default.device