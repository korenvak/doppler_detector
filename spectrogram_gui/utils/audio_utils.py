import librosa
import scipy.signal
import numpy as np

def load_audio_with_filters(path, hp=None, lp=None, gain_db=0):
    y, sr = librosa.load(path, sr=None, mono=True)

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
