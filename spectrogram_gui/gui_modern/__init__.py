"""
Modern Spectrogram GUI Package
High-performance audio analysis with glassmorphic UI
"""

from .main_window_complete import CompleteModernMainWindow
from .spectrogram_canvas import OptimizedSpectrogramCanvas
from .audio_processor import AudioProcessor
from .audio_player import ModernAudioPlayer, AudioDeviceManager

__version__ = "2.0.0"
__all__ = [
    "CompleteModernMainWindow",
    "OptimizedSpectrogramCanvas", 
    "AudioProcessor",
    "ModernAudioPlayer",
    "AudioDeviceManager"
]