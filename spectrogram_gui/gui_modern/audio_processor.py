"""
Audio processing module with performance optimizations
Handles audio loading, filtering, and spectrogram computation
"""

import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from scipy.signal import butter, filtfilt, savgol_filter
from functools import lru_cache
from numba import jit, prange
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


class AudioProcessor:
    """
    High-performance audio processor with caching and optimization
    """
    
    def __init__(self):
        self.cache = {}
        self.sample_rate = None
        self.audio_data = None
        self.filtered_data = None
        
    @staticmethod
    @lru_cache(maxsize=32)
    def load_audio(file_path, target_sr=None):
        """
        Load audio file with caching
        
        Args:
            file_path: Path to audio file
            target_sr: Target sample rate (None to keep original)
            
        Returns:
            audio_data, sample_rate
        """
        try:
            # Try soundfile first (faster for WAV/FLAC)
            audio_data, sample_rate = sf.read(file_path, dtype='float32')
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                
            # Resample if needed
            if target_sr and sample_rate != target_sr:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=target_sr,
                    res_type='kaiser_fast'
                )
                sample_rate = target_sr
                
            return audio_data, sample_rate
            
        except Exception as e:
            # Fallback to librosa
            try:
                audio_data, sample_rate = librosa.load(
                    file_path, 
                    sr=target_sr, 
                    mono=True
                )
                return audio_data, sample_rate
            except Exception as e2:
                raise Exception(f"Failed to load audio: {e2}")
                
    @staticmethod
    @jit(nopython=True, cache=True)
    def apply_gain_fast(audio_data, gain_db):
        """
        Apply gain to audio data (JIT compiled)
        """
        gain_linear = 10.0 ** (gain_db / 20.0)
        return audio_data * gain_linear
        
    def apply_filters(self, audio_data, sample_rate, filters_config):
        """
        Apply multiple filters to audio data
        
        Args:
            audio_data: Input audio
            sample_rate: Sample rate
            filters_config: Dictionary of filter configurations
            
        Returns:
            Filtered audio data
        """
        filtered = audio_data.copy()
        
        # Apply gain
        if 'gain' in filters_config:
            filtered = self.apply_gain_fast(filtered, filters_config['gain'])
            
        # Apply high-pass filter
        if 'highpass' in filters_config:
            cutoff = filters_config['highpass']['cutoff']
            order = filters_config['highpass'].get('order', 5)
            
            sos = signal.butter(order, cutoff, 'hp', fs=sample_rate, output='sos')
            filtered = signal.sosfiltfilt(sos, filtered)
            
        # Apply low-pass filter
        if 'lowpass' in filters_config:
            cutoff = filters_config['lowpass']['cutoff']
            order = filters_config['lowpass'].get('order', 5)
            
            sos = signal.butter(order, cutoff, 'lp', fs=sample_rate, output='sos')
            filtered = signal.sosfiltfilt(sos, filtered)
            
        # Apply band-pass filter
        if 'bandpass' in filters_config:
            low = filters_config['bandpass']['low']
            high = filters_config['bandpass']['high']
            order = filters_config['bandpass'].get('order', 5)
            
            sos = signal.butter(order, [low, high], 'bp', fs=sample_rate, output='sos')
            filtered = signal.sosfiltfilt(sos, filtered)
            
        # Apply notch filter
        if 'notch' in filters_config:
            freq = filters_config['notch']['freq']
            Q = filters_config['notch'].get('Q', 30)
            
            b, a = signal.iirnotch(freq, Q, sample_rate)
            filtered = filtfilt(b, a, filtered)
            
        # Apply smoothing
        if 'smooth' in filters_config:
            window_length = filters_config['smooth'].get('window', 51)
            polyorder = filters_config['smooth'].get('order', 3)
            
            if len(filtered) > window_length:
                filtered = savgol_filter(filtered, window_length, polyorder)
                
        # Normalize if requested
        if filters_config.get('normalize', False):
            max_val = np.max(np.abs(filtered))
            if max_val > 0:
                filtered = filtered / max_val
                
        return filtered
        
    @staticmethod
    @lru_cache(maxsize=16)
    def compute_spectrogram_cached(audio_hash, sample_rate, nperseg, noverlap, window):
        """
        Cached spectrogram computation (for repeated calls with same parameters)
        Note: audio_hash should be a hash of the audio data
        """
        # This is a placeholder - actual implementation would retrieve audio from hash
        pass
        
    @staticmethod
    def compute_spectrogram(audio_data, sample_rate, nperseg=2048, noverlap=None, 
                          window='hann', nfft=None, scaling='density',
                          mode='magnitude'):
        """
        Compute spectrogram with various options
        
        Args:
            audio_data: Input audio signal
            sample_rate: Sample rate
            nperseg: Length of each segment
            noverlap: Number of points to overlap (None = nperseg // 2)
            window: Window function
            nfft: FFT length (None = nperseg)
            scaling: 'density' or 'spectrum'
            mode: 'magnitude', 'power', 'phase', 'complex'
            
        Returns:
            freqs, times, Sxx
        """
        if noverlap is None:
            noverlap = nperseg // 2
            
        if nfft is None:
            nfft = nperseg
            
        # Compute spectrogram
        freqs, times, Sxx = signal.spectrogram(
            audio_data,
            fs=sample_rate,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            scaling=scaling,
            mode='complex',
            detrend='constant'
        )
        
        # Convert to requested mode
        if mode == 'magnitude':
            Sxx = np.abs(Sxx)
        elif mode == 'power':
            Sxx = np.abs(Sxx) ** 2
        elif mode == 'phase':
            Sxx = np.angle(Sxx)
        # elif mode == 'complex': keep as is
        
        return freqs, times, Sxx
        
    @staticmethod
    def compute_mel_spectrogram(audio_data, sample_rate, n_mels=128, 
                               n_fft=2048, hop_length=512, fmin=0, fmax=None):
        """
        Compute mel-scaled spectrogram
        
        Args:
            audio_data: Input audio
            sample_rate: Sample rate
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length
            fmin: Minimum frequency
            fmax: Maximum frequency (None = sr/2)
            
        Returns:
            mel_freqs, times, mel_spec
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax or sample_rate/2,
            power=2.0
        )
        
        # Get time axis
        times = librosa.frames_to_time(
            np.arange(mel_spec.shape[1]),
            sr=sample_rate,
            hop_length=hop_length
        )
        
        # Get mel frequencies
        mel_freqs = librosa.mel_frequencies(
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax or sample_rate/2
        )
        
        return mel_freqs, times, mel_spec
        
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def compute_stft_fast(audio_data, n_fft, hop_length, window):
        """
        Fast STFT computation using Numba
        """
        n_frames = (len(audio_data) - n_fft) // hop_length + 1
        stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
        
        for i in prange(n_frames):
            start = i * hop_length
            frame = audio_data[start:start + n_fft] * window
            fft_result = np.fft.rfft(frame)
            stft_matrix[:, i] = fft_result
            
        return stft_matrix
        
    def compute_chromagram(self, audio_data, sample_rate, hop_length=512, n_chroma=12):
        """
        Compute chromagram for harmonic analysis
        """
        chromagram = librosa.feature.chroma_stft(
            y=audio_data,
            sr=sample_rate,
            hop_length=hop_length,
            n_chroma=n_chroma
        )
        
        times = librosa.frames_to_time(
            np.arange(chromagram.shape[1]),
            sr=sample_rate,
            hop_length=hop_length
        )
        
        return times, chromagram
        
    def detect_onset(self, audio_data, sample_rate, hop_length=512):
        """
        Detect onset times in audio
        """
        onset_frames = librosa.onset.onset_detect(
            y=audio_data,
            sr=sample_rate,
            hop_length=hop_length,
            backtrack=True
        )
        
        onset_times = librosa.frames_to_time(
            onset_frames,
            sr=sample_rate,
            hop_length=hop_length
        )
        
        return onset_times
        
    def extract_features(self, audio_data, sample_rate):
        """
        Extract various audio features for analysis
        """
        features = {}
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio_data, sr=sample_rate
        )[0]
        features['spectral_centroid'] = np.mean(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_data, sr=sample_rate
        )[0]
        features['spectral_rolloff'] = np.mean(spectral_rolloff)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features['zero_crossing_rate'] = np.mean(zcr)
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        features['mfcc'] = np.mean(mfccs, axis=1)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
        features['tempo'] = tempo
        
        return features