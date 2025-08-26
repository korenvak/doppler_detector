import numpy as np
import librosa
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')


@dataclass
class DopplerEvent:
    """Represents a detected Doppler event (aircraft/drone flyover)"""
    start_time: float  # seconds
    end_time: float  # seconds
    tracks: List[List[Tuple[float, float]]]  # List of (time, freq) tracks
    confidence: float
    event_type: str  # 'drone', 'aircraft', 'unknown'
    closest_approach_time: float  # seconds - peak amplitude time
    frequency_range: Tuple[float, float]  # Min/max frequencies in Hz
    doppler_signature: Dict  # Physical characteristics
    amplitude_envelope: np.ndarray  # Power vs time


class PhysicalDopplerDetector:
    """
    Physics-based Doppler event detector for aircraft/drone detection.

    Uses realistic aeroacoustic models and Doppler physics to identify
    genuine aircraft/drone flyovers while rejecting false positives.
    """

    def __init__(
            self,
            # Aircraft/drone parameters
            min_altitude_m: float = 30.0,
            max_altitude_m: float = 1000.0,
            min_speed_ms: float = 5.0,  # m/s - minimum aircraft speed
            max_speed_ms: float = 100.0,  # m/s - maximum for small aircraft/drones

            # BPF detection parameters
            min_bpf_hz: float = 50.0,  # Minimum blade passage frequency
            max_bpf_hz: float = 500.0,  # Maximum BPF for small aircraft
            bpf_tolerance_hz: float = 15.0,  # BPF tracking tolerance

            # Event characteristics
            min_event_duration_s: float = 8.0,  # Minimum realistic flyover
            max_event_duration_s: float = 180.0,  # Maximum time in detection range
            min_snr_db: float = 8.0,  # Minimum signal-to-noise ratio

            # Doppler pattern validation
            max_doppler_rate_hz_s: float = 8.0,  # Maximum realistic Doppler rate
            min_doppler_shift_hz: float = 2.0,  # Minimum detectable shift
            continuity_threshold: float = 0.7,  # Track continuity requirement

            # Amplitude envelope validation
            envelope_rise_time_s: float = 5.0,  # Time to reach peak (approach)
            envelope_fall_time_s: float = 5.0,  # Time from peak (recede)
            peak_prominence_factor: float = 2.0,  # Peak must be X times background
    ):
        # Store all parameters
        self.min_altitude_m = min_altitude_m
        self.max_altitude_m = max_altitude_m
        self.min_speed_ms = min_speed_ms
        self.max_speed_ms = max_speed_ms

        self.min_bpf_hz = min_bpf_hz
        self.max_bpf_hz = max_bpf_hz
        self.bpf_tolerance_hz = bpf_tolerance_hz

        self.min_event_duration_s = min_event_duration_s
        self.max_event_duration_s = max_event_duration_s
        self.min_snr_db = min_snr_db

        self.max_doppler_rate_hz_s = max_doppler_rate_hz_s
        self.min_doppler_shift_hz = min_doppler_shift_hz
        self.continuity_threshold = continuity_threshold

        self.envelope_rise_time_s = envelope_rise_time_s
        self.envelope_fall_time_s = envelope_fall_time_s
        self.peak_prominence_factor = peak_prominence_factor

        # Speed of sound (will be refined based on conditions)
        self.c_sound = 343.0  # m/s at 20°C

    def detect_events_from_file(self, audio_file_path: str) -> List[DopplerEvent]:
        """
        Main entry point - detect Doppler events from audio file.

        Args:
            audio_file_path: Path to FLAC/WAV audio file

        Returns:
            List of detected DopplerEvent objects
        """
        print(f"[Physical Doppler] Loading audio from: {audio_file_path}")

        # Load audio
        y, sr = librosa.load(audio_file_path, sr=None, mono=True)
        duration = len(y) / sr
        print(f"[Physical Doppler] Loaded {duration:.1f}s of audio at {sr}Hz")

        # Compute spectrogram optimized for aircraft detection
        freqs, times, Sxx = self._compute_aircraft_spectrogram(y, sr)
        print(f"[Physical Doppler] Spectrogram: {len(freqs)} freq bins, {len(times)} time frames")

        # Detect events
        return self.detect_events(Sxx, freqs, times)

    def detect_events(self, Sxx: np.ndarray, freqs: np.ndarray, times: np.ndarray) -> List[DopplerEvent]:
        """
        Detect Doppler events in spectrogram data using physics-based approach.

        Args:
            Sxx: Spectrogram power data (freq x time)
            freqs: Frequency array in Hz
            times: Time array in seconds

        Returns:
            List of detected DopplerEvent objects
        """
        print(f"[Physical Doppler] Starting detection on {Sxx.shape[0]}×{Sxx.shape[1]} spectrogram")

        # Step 1: Find candidate BPF tracks
        bpf_candidates = self._find_bpf_candidates(Sxx, freqs, times)
        print(f"[Physical Doppler] Found {len(bpf_candidates)} BPF candidates")

        # Step 2: Validate Doppler patterns
        doppler_tracks = self._validate_doppler_patterns(bpf_candidates, freqs, times)
        print(f"[Physical Doppler] {len(doppler_tracks)} tracks show Doppler patterns")

        # Step 3: Group tracks into events (handle interruptions)
        event_groups = self._group_tracks_into_events(doppler_tracks, times)
        print(f"[Physical Doppler] Grouped into {len(event_groups)} potential events")

        # Step 4: Validate event physics
        valid_events = self._validate_event_physics(event_groups, Sxx, freqs, times)
        print(f"[Physical Doppler] {len(valid_events)} events pass physics validation")

        # Step 5: Extract event characteristics
        final_events = self._extract_event_characteristics(valid_events, Sxx, freqs, times)
        print(f"[Physical Doppler] Detection complete: {len(final_events)} events")

        return final_events

    def _compute_aircraft_spectrogram(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram optimized for aircraft/drone detection.
        Uses parameters that enhance BPF and harmonic visibility.
        """
        # Parameters optimized for aircraft detection
        n_fft = 4096  # High frequency resolution for BPF tracking
        hop_length = n_fft // 8  # Good time resolution for Doppler tracking
        window = 'blackmanharris'  # Low side-lobe window for clean harmonics

        # Compute STFT
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
        Sxx = np.abs(stft) ** 2  # Power spectrogram

        # Convert to dB with noise floor
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        # Normalize to [0, 1] for consistent thresholding
        Sxx_norm = (Sxx_db - np.min(Sxx_db)) / (np.max(Sxx_db) - np.min(Sxx_db))

        # Generate frequency and time arrays
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=hop_length)

        return freqs, times, Sxx_norm

    def _find_bpf_candidates(self, Sxx: np.ndarray, freqs: np.ndarray, times: np.ndarray) -> List[
        List[Tuple[float, float, float]]]:
        """
        Find potential Blade Passage Frequency tracks.

        Returns:
            List of tracks, each track is list of (time, frequency, amplitude) tuples
        """
        # Focus on BPF frequency range
        freq_mask = (freqs >= self.min_bpf_hz) & (freqs <= self.max_bpf_hz)
        freq_indices = np.where(freq_mask)[0]

        if len(freq_indices) == 0:
            return []

        freq_start, freq_end = freq_indices[0], freq_indices[-1]

        # Detect peaks in each time frame
        tracks = []
        active_tracks = []  # (last_time_idx, last_freq_idx, track_data)

        for t_idx, time in enumerate(times):
            # Extract frequency slice at this time
            power_slice = Sxx[freq_start:freq_end + 1, t_idx]

            # Find peaks with adaptive threshold
            # Use local median as baseline to handle varying noise floor
            local_baseline = np.median(power_slice)
            threshold = local_baseline + 0.1  # Adaptive threshold above baseline

            peak_indices, properties = find_peaks(
                power_slice,
                height=threshold,
                prominence=0.05,
                distance=5  # Minimum separation between peaks
            )

            if len(peak_indices) == 0:
                continue

            # Convert to absolute frequency indices and get amplitudes
            abs_freq_indices = peak_indices + freq_start
            amplitudes = properties['peak_heights']

            # Sort by amplitude (strongest peaks first)
            sort_order = np.argsort(amplitudes)[::-1]
            abs_freq_indices = abs_freq_indices[sort_order]
            amplitudes = amplitudes[sort_order]

            # Try to continue existing tracks
            used_peaks = set()
            new_active_tracks = []

            for last_t_idx, last_f_idx, track_data in active_tracks:
                last_freq = freqs[last_f_idx]
                extended = False

                # Find best matching peak within tolerance
                for peak_idx, (f_idx, amp) in enumerate(zip(abs_freq_indices, amplitudes)):
                    if peak_idx in used_peaks:
                        continue

                    current_freq = freqs[f_idx]
                    freq_diff = abs(current_freq - last_freq)

                    if freq_diff <= self.bpf_tolerance_hz:
                        # Continue this track
                        used_peaks.add(peak_idx)
                        track_data.append((time, current_freq, amp))
                        new_active_tracks.append((t_idx, f_idx, track_data))
                        extended = True
                        break

                if not extended:
                    # Track ended - add to completed tracks if long enough
                    if len(track_data) >= int(self.min_event_duration_s / (times[1] - times[0])) // 4:
                        tracks.append(track_data)

            # Start new tracks from unused peaks
            for peak_idx, (f_idx, amp) in enumerate(zip(abs_freq_indices, amplitudes)):
                if peak_idx not in used_peaks:
                    new_track_data = [(time, freqs[f_idx], amp)]
                    new_active_tracks.append((t_idx, f_idx, new_track_data))

            active_tracks = new_active_tracks

        # Add remaining active tracks
        for _, _, track_data in active_tracks:
            if len(track_data) >= int(self.min_event_duration_s / (times[1] - times[0])) // 4:
                tracks.append(track_data)

        return tracks

    def _validate_doppler_patterns(self, bpf_candidates: List[List[Tuple[float, float, float]]],
                                   freqs: np.ndarray, times: np.ndarray) -> List[List[Tuple[float, float, float]]]:
        """
        Validate that BPF candidates show realistic Doppler shift patterns.
        """
        valid_tracks = []

        for track in bpf_candidates:
            if len(track) < 10:  # Need minimum points for Doppler analysis
                continue

            # Extract time and frequency arrays
            track_times = np.array([point[0] for point in track])
            track_freqs = np.array([point[1] for point in track])
            track_amps = np.array([point[2] for point in track])

            # Smooth frequency track to reduce noise
            if len(track_freqs) > 5:
                try:
                    # Use Savitzky-Golay filter for smooth Doppler curve
                    window_length = min(11, len(track_freqs) // 2 * 2 + 1)  # Ensure odd
                    if window_length >= 5:
                        track_freqs_smooth = savgol_filter(track_freqs, window_length, 3)
                    else:
                        track_freqs_smooth = gaussian_filter1d(track_freqs, sigma=1.0)
                except:
                    track_freqs_smooth = gaussian_filter1d(track_freqs, sigma=1.0)
            else:
                track_freqs_smooth = track_freqs

            # Calculate Doppler rate (df/dt)
            if len(track_times) > 2:
                doppler_rates = np.gradient(track_freqs_smooth, track_times)
                max_doppler_rate = np.max(np.abs(doppler_rates))

                # Check if Doppler rate is within realistic bounds
                if max_doppler_rate > self.max_doppler_rate_hz_s:
                    continue  # Too fast to be realistic

                # Check for minimum Doppler shift
                freq_range = np.max(track_freqs) - np.min(track_freqs)
                if freq_range < self.min_doppler_shift_hz:
                    continue  # No significant Doppler shift

            # Check track continuity (no big jumps)
            freq_diffs = np.abs(np.diff(track_freqs_smooth))
            dt = np.mean(np.diff(track_times))
            max_jump = np.max(freq_diffs)

            if max_jump > self.bpf_tolerance_hz * 2:  # Allow some flexibility
                continue  # Too jumpy

            # Check for realistic Doppler curve shape
            if self._validate_doppler_curve_shape(track_times, track_freqs_smooth):
                # Update track with smoothed frequencies
                smooth_track = [(t, f_smooth, a) for t, f_smooth, a in
                                zip(track_times, track_freqs_smooth, track_amps)]
                valid_tracks.append(smooth_track)

        return valid_tracks

    def _validate_doppler_curve_shape(self, times: np.ndarray, freqs: np.ndarray) -> bool:
        """
        Check if the frequency vs time curve looks like a realistic Doppler shift.
        Should show approach (increasing freq) then recede (decreasing freq) or vice versa.
        """
        if len(freqs) < 6:
            return False

        # Find the peak/valley in frequency (closest approach)
        # Smooth first to avoid noise
        freqs_smooth = gaussian_filter1d(freqs, sigma=1.0)

        # Look for a clear maximum or minimum (closest approach point)
        mid_start = len(freqs) // 4
        mid_end = 3 * len(freqs) // 4

        # Check for frequency peak (approaching then receding)
        max_idx = np.argmax(freqs_smooth[mid_start:mid_end]) + mid_start

        # Check for frequency valley (receding then approaching)
        min_idx = np.argmin(freqs_smooth[mid_start:mid_end]) + mid_start

        # Analyze slope before and after peak/valley
        def analyze_slopes(peak_idx):
            if peak_idx < 3 or peak_idx > len(freqs) - 3:
                return False

            # Before peak
            before_slope = np.polyfit(times[:peak_idx], freqs_smooth[:peak_idx], 1)[0] if peak_idx > 1 else 0
            # After peak
            after_slope = np.polyfit(times[peak_idx:], freqs_smooth[peak_idx:], 1)[0] if peak_idx < len(
                freqs) - 1 else 0

            # For frequency maximum: should increase then decrease
            # For frequency minimum: should decrease then increase
            return abs(before_slope) > 0.1 and abs(after_slope) > 0.1 and np.sign(before_slope) != np.sign(after_slope)

        return analyze_slopes(max_idx) or analyze_slopes(min_idx)

    def _group_tracks_into_events(self, doppler_tracks: List[List[Tuple[float, float, float]]],
                                  times: np.ndarray) -> List[List[List[Tuple[float, float, float]]]]:
        """
        Group related tracks into events, handling interruptions due to range/masking.
        Aircraft may leave and re-enter detection range, creating multiple tracks for one event.
        """
        if not doppler_tracks:
            return []

        # Sort tracks by start time
        sorted_tracks = sorted(doppler_tracks, key=lambda track: track[0][0])

        events = []
        current_event = [sorted_tracks[0]]

        for i in range(1, len(sorted_tracks)):
            current_track = sorted_tracks[i]
            last_track_in_event = current_event[-1]

            # Get timing information
            current_start = current_track[0][0]
            last_end = last_track_in_event[-1][0]

            # Get frequency information
            current_freq_avg = np.mean([point[1] for point in current_track])
            last_freq_avg = np.mean([point[1] for point in last_track_in_event])

            # Calculate time gap and frequency similarity
            time_gap = current_start - last_end
            freq_diff = abs(current_freq_avg - last_freq_avg)

            # Decision criteria for grouping
            max_time_gap = 30.0  # seconds - aircraft can be out of range briefly
            max_freq_diff = 50.0  # Hz - frequency can shift due to different harmonics

            if time_gap <= max_time_gap and freq_diff <= max_freq_diff:
                # Likely same aircraft - add to current event
                current_event.append(current_track)
            else:
                # Different aircraft or too much gap - start new event
                if len(current_event) > 0:
                    events.append(current_event)
                current_event = [current_track]

        # Add last event
        if len(current_event) > 0:
            events.append(current_event)

        return events

    def _validate_event_physics(self, event_groups: List[List[List[Tuple[float, float, float]]]],
                                Sxx: np.ndarray, freqs: np.ndarray, times: np.ndarray) -> List[
        List[List[Tuple[float, float, float]]]]:
        """
        Apply physics-based validation to filter out non-aircraft events.
        """
        valid_events = []

        for event_group in event_groups:
            # Combine all tracks in event for analysis
            all_points = []
            for track in event_group:
                all_points.extend(track)

            if len(all_points) < 10:
                continue

            # Sort by time
            all_points.sort(key=lambda x: x[0])

            # Extract arrays
            event_times = np.array([point[0] for point in all_points])
            event_freqs = np.array([point[1] for point in all_points])
            event_amps = np.array([point[2] for point in all_points])

            # Check event duration
            duration = event_times[-1] - event_times[0]
            if duration < self.min_event_duration_s or duration > self.max_event_duration_s:
                continue

            # Check SNR
            signal_power = np.mean(event_amps)
            noise_estimate = self._estimate_noise_floor(Sxx, freqs, times, event_times, event_freqs)
            snr_db = 10 * np.log10(signal_power / (noise_estimate + 1e-10))

            if snr_db < self.min_snr_db:
                continue

            # Check amplitude envelope shape (should show approach and recede)
            if not self._validate_amplitude_envelope(event_times, event_amps):
                continue

            # Physics check: estimate realistic altitude and speed
            if self._validate_aircraft_physics(event_times, event_freqs, event_amps):
                valid_events.append(event_group)

        return valid_events

    def _estimate_noise_floor(self, Sxx: np.ndarray, freqs: np.ndarray, times: np.ndarray,
                              event_times: np.ndarray, event_freqs: np.ndarray) -> float:
        """
        Estimate noise floor around the event for SNR calculation.
        """
        # Find frequency and time indices for the event
        freq_min, freq_max = np.min(event_freqs), np.max(event_freqs)
        time_min, time_max = np.min(event_times), np.max(event_times)

        # Expand region slightly
        freq_margin = (freq_max - freq_min) * 0.5
        time_margin = (time_max - time_min) * 0.2

        # Find background regions
        freq_mask_bg = ((freqs >= freq_min - freq_margin) & (freqs <= freq_min)) | \
                       ((freqs >= freq_max) & (freqs <= freq_max + freq_margin))
        time_mask_bg = ((times >= time_min - time_margin) & (times <= time_min)) | \
                       ((times >= time_max) & (times <= time_max + time_margin))

        if not np.any(freq_mask_bg) or not np.any(time_mask_bg):
            return np.median(Sxx) * 0.1  # Fallback

        # Sample background
        bg_power = Sxx[np.ix_(freq_mask_bg, time_mask_bg)]
        return np.median(bg_power)

    def _validate_amplitude_envelope(self, times: np.ndarray, amplitudes: np.ndarray) -> bool:
        """
        Check if amplitude envelope shows realistic aircraft approach/recede pattern.
        """
        if len(amplitudes) < 6:
            return False

        # Smooth amplitude envelope
        smooth_amps = gaussian_filter1d(amplitudes, sigma=1.0)

        # Find peak
        peak_idx = np.argmax(smooth_amps)
        peak_time = times[peak_idx]
        peak_amp = smooth_amps[peak_idx]

        # Check that peak is significantly above baseline
        baseline = np.median(smooth_amps)
        if peak_amp < baseline * self.peak_prominence_factor:
            return False

        # Check rise and fall pattern
        if peak_idx > 0:
            rise_amps = smooth_amps[:peak_idx]
            if len(rise_amps) > 2:
                rise_trend = np.polyfit(range(len(rise_amps)), rise_amps, 1)[0]
                if rise_trend < 0:  # Should be increasing towards peak
                    return False

        if peak_idx < len(smooth_amps) - 1:
            fall_amps = smooth_amps[peak_idx:]
            if len(fall_amps) > 2:
                fall_trend = np.polyfit(range(len(fall_amps)), fall_amps, 1)[0]
                if fall_trend > 0:  # Should be decreasing from peak
                    return False

        return True

    def _validate_aircraft_physics(self, times: np.ndarray, freqs: np.ndarray, amplitudes: np.ndarray) -> bool:
        """
        Check if the event parameters are consistent with realistic aircraft physics.
        """
        # Estimate aircraft parameters from Doppler signature
        try:
            # Find closest approach time (peak amplitude)
            peak_idx = np.argmax(amplitudes)
            peak_time = times[peak_idx]

            # Estimate base frequency (at closest approach)
            base_freq = freqs[peak_idx]

            # Look at frequency change rate around peak
            window = 5
            start_idx = max(0, peak_idx - window)
            end_idx = min(len(freqs), peak_idx + window + 1)

            if end_idx - start_idx < 3:
                return True  # Too short to analyze, give benefit of doubt

            # Estimate maximum Doppler shift
            freq_window = freqs[start_idx:end_idx]
            max_doppler_shift = np.max(freq_window) - np.min(freq_window)

            # Estimate speed using Doppler formula: Δf = (v/c) * f0
            # For small angles, maximum shift occurs at closest approach
            if max_doppler_shift > 0:
                estimated_speed = (max_doppler_shift * self.c_sound) / base_freq

                # Check if speed is realistic
                if estimated_speed < self.min_speed_ms or estimated_speed > self.max_speed_ms:
                    return False

            # Event duration check vs altitude
            duration = times[-1] - times[0]

            # For a given speed and altitude, there's a relationship with detection duration
            # This is a rough check - real physics is more complex
            min_expected_duration = 2 * self.min_altitude_m / self.max_speed_ms
            max_expected_duration = 2 * self.max_altitude_m / self.min_speed_ms

            if duration < min_expected_duration * 0.5 or duration > max_expected_duration * 2.0:
                return False

            return True

        except Exception:
            # If physics analysis fails, err on side of caution
            return True

    def _extract_event_characteristics(self, valid_events: List[List[List[Tuple[float, float, float]]]],
                                       Sxx: np.ndarray, freqs: np.ndarray, times: np.ndarray) -> List[DopplerEvent]:
        """
        Extract detailed characteristics from validated events.
        """
        doppler_events = []

        for event_idx, event_group in enumerate(valid_events):
            # Combine all tracks
            all_points = []
            for track in event_group:
                all_points.extend(track)
            all_points.sort(key=lambda x: x[0])

            if len(all_points) < 5:
                continue

            # Extract basic info
            event_times = np.array([point[0] for point in all_points])
            event_freqs = np.array([point[1] for point in all_points])
            event_amps = np.array([point[2] for point in all_points])

            start_time = np.min(event_times)
            end_time = np.max(event_times)
            peak_idx = np.argmax(event_amps)
            closest_approach_time = event_times[peak_idx]

            # Classify event type based on frequency range and behavior
            base_freq = np.median(event_freqs)
            if base_freq < 150:
                event_type = 'aircraft'
            elif base_freq < 300:
                event_type = 'drone'
            else:
                event_type = 'unknown'

            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(event_times, event_freqs, event_amps)

            # Create tracks in the format expected by the GUI
            # Convert each original track to (time, frequency) pairs
            tracks_for_gui = []
            for track in event_group:
                gui_track = [(point[0], point[1]) for point in track]
                tracks_for_gui.append(gui_track)

            # Extract Doppler signature
            doppler_signature = self._extract_doppler_signature(event_times, event_freqs, event_amps)

            # Create event object
            doppler_event = DopplerEvent(
                start_time=start_time,
                end_time=end_time,
                tracks=tracks_for_gui,
                confidence=confidence,
                event_type=event_type,
                closest_approach_time=closest_approach_time,
                frequency_range=(np.min(event_freqs), np.max(event_freqs)),
                doppler_signature=doppler_signature,
                amplitude_envelope=event_amps
            )

            doppler_events.append(doppler_event)

        return doppler_events

    def _calculate_confidence(self, times: np.ndarray, freqs: np.ndarray, amplitudes: np.ndarray) -> float:
        """
        Calculate confidence score for the detection.
        """
        confidence = 0.0

        # Duration factor (longer is more confident)
        duration = times[-1] - times[0]
        duration_score = min(duration / 30.0, 1.0) * 0.2
        confidence += duration_score

        # Amplitude consistency (stable signal is more confident)
        amp_std = np.std(amplitudes) / (np.mean(amplitudes) + 1e-10)
        amp_score = max(0, 1.0 - amp_std / 0.5) * 0.25
        confidence += amp_score

        # Frequency smoothness (smooth Doppler curve is more confident)
        if len(freqs) > 2:
            freq_smoothness = np.std(np.diff(freqs))
            smooth_score = max(0, 1.0 - freq_smoothness / 10.0) * 0.2
            confidence += smooth_score

        # SNR proxy (higher amplitude relative to baseline)
        baseline_amp = np.percentile(amplitudes, 20)
        peak_amp = np.max(amplitudes)
        snr_proxy = peak_amp / (baseline_amp + 1e-10)
        snr_score = min(snr_proxy / 5.0, 1.0) * 0.2
        confidence += snr_score

        # Doppler shift magnitude (clear shift is more confident)
        freq_range = np.max(freqs) - np.min(freqs)
        doppler_score = min(freq_range / 20.0, 1.0) * 0.15
        confidence += doppler_score

        return min(confidence, 1.0)

    def _extract_doppler_signature(self, times: np.ndarray, freqs: np.ndarray, amplitudes: np.ndarray) -> Dict:
        """
        Extract detailed Doppler signature characteristics.
        """
        signature = {}

        # Basic statistics
        signature['duration'] = times[-1] - times[0]
        signature['peak_frequency'] = freqs[np.argmax(amplitudes)]
        signature['frequency_span'] = np.max(freqs) - np.min(freqs)
        signature['peak_amplitude'] = np.max(amplitudes)

        # Doppler characteristics
        if len(freqs) > 3:
            # Estimate maximum Doppler rate
            doppler_rates = np.abs(np.gradient(freqs, times))
            signature['max_doppler_rate'] = np.max(doppler_rates)
            signature['avg_doppler_rate'] = np.mean(doppler_rates)

            # Classify Doppler pattern
            peak_time_idx = np.argmax(amplitudes)
            if peak_time_idx < len(freqs) // 3:
                signature['pattern'] = 'recede'  # Peak early, mostly receding
            elif peak_time_idx > 2 * len(freqs) // 3:
                signature['pattern'] = 'approach'  # Peak late, mostly approaching
            else:
                signature['pattern'] = 'flyover'  # Peak in middle, approach + recede
        else:
            signature['max_doppler_rate'] = 0.0
            signature['avg_doppler_rate'] = 0.0
            signature['pattern'] = 'unknown'

        # Estimate physical parameters
        base_freq = np.median(freqs)
        max_shift = signature['frequency_span'] / 2.0

        if max_shift > 0:
            # Rough speed estimate using Doppler formula
            estimated_speed = (max_shift * self.c_sound) / base_freq
            signature['estimated_speed_ms'] = min(estimated_speed, self.max_speed_ms)

            # Rough altitude estimate (very approximate)
            duration = signature['duration']
            signature['estimated_altitude_m'] = min(estimated_speed * duration / 4.0, self.max_altitude_m)
        else:
            signature['estimated_speed_ms'] = 0.0
            signature['estimated_altitude_m'] = 0.0

        return signature


def main():
    """
    Example usage and testing function.
    """
    # Create detector with default parameters
    detector = PhysicalDopplerDetector(
        min_event_duration_s=10.0,
        max_event_duration_s=120.0,
        min_snr_db=6.0,
        max_doppler_rate_hz_s=5.0
    )

    print("Physical Doppler Detector initialized")
    print(f"Detection parameters:")
    print(f"  Event duration: {detector.min_event_duration_s}-{detector.max_event_duration_s}s")
    print(f"  BPF range: {detector.min_bpf_hz}-{detector.max_bpf_hz} Hz")
    print(f"  Speed range: {detector.min_speed_ms}-{detector.max_speed_ms} m/s")
    print(f"  Min SNR: {detector.min_snr_db} dB")

    # Example of how to use with audio file
    # events = detector.detect_events_from_file("path/to/audio.flac")
    #
    # for i, event in enumerate(events):
    #     print(f"\nEvent {i+1}:")
    #     print(f"  Time: {event.start_time:.1f}-{event.end_time:.1f}s")
    #     print(f"  Type: {event.event_type}")
    #     print(f"  Confidence: {event.confidence:.2f}")
    #     print(f"  Frequency range: {event.frequency_range[0]:.1f}-{event.frequency_range[1]:.1f} Hz")
    #     print(f"  Tracks: {len(event.tracks)}")
    #     print(f"  Doppler pattern: {event.doppler_signature.get('pattern', 'unknown')}")
    #     print(f"  Estimated speed: {event.doppler_signature.get('estimated_speed_ms', 0):.1f} m/s")


if __name__ == "__main__":
    main()