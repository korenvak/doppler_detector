import librosa
import soundfile as sf
import scipy.signal
import numpy as np
from doppler_detector.spectrogram_gui.utils.logger import debug, info, warning, error, timer

def load_audio_with_filters(path, hp=None, lp=None, gain_db=0):
    """Load audio from ``path`` and optionally apply simple filters."""
    debug(f"Loading audio file: {path}")
    with timer(f"Audio loading from {path}"):
        try:
            y, sr = librosa.load(path, sr=None, mono=True)
            debug(f"Loaded {len(y)} samples at {sr}Hz using librosa")
        except Exception as e:
            msg = str(e).lower()
            if "array is too big" in msg or "psf_fseek" in msg:
                warning(f"Large file detected, using streaming read: {path}")
                # Fallback to a streaming read with soundfile to avoid large
                # allocations or libsndfile seek errors on huge FLAC files
                with sf.SoundFile(path) as f:
                    sr = f.samplerate
                    blocks = []
                    while True:
                        data = f.read(65536)
                        if not len(data):
                            break
                        if f.channels > 1:
                            data = np.mean(data, axis=1)
                        blocks.append(data)
                    y = np.concatenate(blocks) if blocks else np.array([], dtype=np.float32)
                debug(f"Streamed {len(y)} samples at {sr}Hz using soundfile")
            else:
                error(f"Failed to load audio: {e}")
                raise

    # High-pass filter
    if hp:
        sos = scipy.signal.butter(4, hp, 'hp', fs=sr, output='sos')
        y = scipy.signal.sosfilt(sos, y)

    # Low-pass filter
    if lp:
        sos = scipy.signal.butter(4, lp, 'lp', fs=sr, output='sos')
        y = scipy.signal.sosfilt(sos, y)

    # Gain (in dB)
    if gain_db:
        factor = 10 ** (gain_db / 20.0)
        y = y * factor

    return y, sr
