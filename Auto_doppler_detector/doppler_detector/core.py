import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import spectrogram, find_peaks
import soundfile as sf
from tqdm import tqdm

# Default spectrogram parameters
DEFAULT_FS = 8000
DEFAULT_NPERSEG = 4096
DEFAULT_NOVERLAP = int(0.8 * DEFAULT_NPERSEG)

# Default frequency range (Hz)
DEFAULT_FREQ_MIN = 100
DEFAULT_FREQ_MAX = 1500

class DopplerDetector:
    def __init__(
        self,
        fs=DEFAULT_FS,
        nperseg=DEFAULT_NPERSEG,
        noverlap=DEFAULT_NOVERLAP,
        freq_min=DEFAULT_FREQ_MIN,
        freq_max=DEFAULT_FREQ_MAX,
        power_threshold=0.2,
        peak_prominence=0.06,
        max_gap_frames=4,
        gap_power_factor=0.8,
        gap_prominence_factor=0.8,
        max_freq_jump_hz=15.0,
        gap_max_jump_hz=10.0,
        max_peaks_per_frame=20,
        min_track_length_frames=13,
        min_track_avg_power=0.1,
        max_track_freq_std_hz=70.0,
        merge_gap_frames=100,
        merge_max_freq_diff_hz=30.0,
        smooth_sigma=1.5,
        median_filter_size=(3, 1)
    ):
        """
        Initialize the DopplerDetector with all tunable parameters.
        Any of these can be modified at runtime (e.g., via a dialog).
        """
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.power_threshold = power_threshold
        self.peak_prominence = peak_prominence
        self.max_gap_frames = max_gap_frames
        self.gap_power_factor = gap_power_factor
        self.gap_prominence_factor = gap_prominence_factor
        self.max_freq_jump_hz = max_freq_jump_hz
        self.gap_max_jump_hz = gap_max_jump_hz
        self.max_peaks_per_frame = max_peaks_per_frame
        self.min_track_length_frames = min_track_length_frames
        self.min_track_avg_power = min_track_avg_power
        self.max_track_freq_std_hz = max_track_freq_std_hz
        self.merge_gap_frames = merge_gap_frames
        self.merge_max_freq_diff_hz = merge_max_freq_diff_hz
        self.smooth_sigma = smooth_sigma
        self.median_filter_size = median_filter_size

        # Containers for spectrogram results
        self.freqs = None       # Frequency bins (Hz)
        self.times = None       # Time bins (s)
        self.Sxx_norm = None    # Normalized spectrogram
        self.Sxx_filt = None    # Filtered spectrogram

    def load_audio(self, filepath):
        """
        Load an audio file (FLAC, WAV, etc.) and return a mono signal array and sample rate.
        If the file has multiple channels, average them to mono.
        """
        y, sr = sf.read(filepath)
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y, sr

    def compute_spectrogram(self, y, sr):
        """
        Compute a magnitude spectrogram (in dB), normalize it to [0, 1],
        then apply a Gaussian filter followed by a median filter.
        Returns:
            freqs: array of frequency bins (Hz)
            times: array of time bins (s)
            Sxx_norm: normalized magnitude spectrogram
            Sxx_filt: filtered spectrogram
        """
        f, t, Sxx = spectrogram(
            y,
            fs=sr,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            window="blackmanharris",
            scaling="density",
            mode="magnitude"
        )
        # Convert to decibels
        Sxx_dB = 10.0 * np.log10(Sxx + 1e-10)
        # Normalize to [0, 1]
        Sxx_norm = (Sxx_dB - Sxx_dB.min()) / (Sxx_dB.max() - Sxx_dB.min())
        # Smooth with a Gaussian filter
        Sxx_smoothed = gaussian_filter(Sxx_norm, sigma=self.smooth_sigma)
        # Further reduce noise with a median filter
        Sxx_filtered = median_filter(Sxx_smoothed, size=self.median_filter_size)

        # Store for later use
        self.freqs = f
        self.times = t
        self.Sxx_norm = Sxx_norm
        self.Sxx_filt = Sxx_filtered
        return f, t, Sxx_norm, Sxx_filtered

    def detect_peaks_per_frame(self):
        """
        For each time column in the filtered spectrogram, find peaks above
        power_threshold and peak_prominence. Keep up to max_peaks_per_frame strongest peaks.
        Returns a list of lists: peaks_per_frame[ti] = [freq_index_1, freq_index_2, ...]
        """
        f = self.freqs
        Sxx_filtered = self.Sxx_filt

        # Determine frequency index range to search in
        f_min_idx = np.searchsorted(f, self.freq_min, side="left")
        f_max_idx = np.searchsorted(f, self.freq_max, side="right") - 1
        f_min_idx = max(f_min_idx, 0)
        f_max_idx = min(f_max_idx, len(f) - 1)

        num_time_bins = Sxx_filtered.shape[1]
        peaks_per_frame = [[] for _ in range(num_time_bins)]

        for ti in tqdm(range(num_time_bins), desc="Detecting peaks per frame"):
            column = Sxx_filtered[f_min_idx : f_max_idx + 1, ti]
            peak_idxs, props = find_peaks(
                column,
                height=self.power_threshold,
                prominence=self.peak_prominence
            )
            if peak_idxs.size > 0:
                heights = props["peak_heights"]
                sorted_indices = np.argsort(heights)[::-1]
                selected = sorted_indices[: self.max_peaks_per_frame]
                absolute_idxs = (peak_idxs[selected] + f_min_idx).tolist()
            else:
                absolute_idxs = []

            peaks_per_frame[ti] = absolute_idxs

        return peaks_per_frame

    def track_peaks_over_time(self, peaks_per_frame):
        """
        Link peaks from frame to frame into “tracks”. Allows gaps up to max_gap_frames
        and uses relaxed criteria within a gap. After linking, filters out tracks
        that are too short, too weak, or too noisy.
        Returns a list of valid tracks, where each track is a list of (time_idx, freq_idx).
        """
        f = self.freqs
        Sxx_filtered = self.Sxx_filt
        num_time_bins = len(peaks_per_frame)

        finished_tracks = []
        active_tracks = []  # Each element: (last_time_idx, last_freq_idx, gap_count, track_points_list)

        for ti in tqdm(range(num_time_bins), desc="Tracking peaks over time"):
            current_peaks = peaks_per_frame[ti]
            used = set()
            new_active = []

            # 1. Try to match existing active tracks
            for last_ti, last_f_idx, gap_count, track_list in active_tracks:
                if gap_count > self.max_gap_frames:
                    finished_tracks.append(track_list)
                    continue

                prev_freq = f[last_f_idx]
                prev_power = Sxx_filtered[last_f_idx, last_ti]

                best_match = None
                best_dist = None

                # Strict match first
                for pi, f_idx in enumerate(current_peaks):
                    if pi in used:
                        continue
                    curr_freq = f[f_idx]
                    dist = abs(curr_freq - prev_freq)
                    if dist <= self.max_freq_jump_hz:
                        curr_power = Sxx_filtered[f_idx, ti]
                        if curr_power >= 0.8 * prev_power:
                            if best_dist is None or dist < best_dist:
                                best_dist = dist
                                best_match = pi

                if best_match is not None:
                    matched_f_idx = current_peaks[best_match]
                    used.add(best_match)
                    new_list = track_list + [(ti, matched_f_idx)]
                    new_active.append((ti, matched_f_idx, 0, new_list))
                    continue

                # Relaxed (gap) match
                lower_freq = prev_freq - self.gap_max_jump_hz
                upper_freq = prev_freq + self.gap_max_jump_hz
                lower_bin = np.searchsorted(f, lower_freq, side="left")
                upper_bin = np.searchsorted(f, upper_freq, side="right") - 1
                lower_bin = max(lower_bin, 0)
                upper_bin = min(upper_bin, len(f) - 1)

                if lower_bin <= upper_bin:
                    segment = Sxx_filtered[lower_bin : upper_bin + 1, ti]
                    pks, pr = find_peaks(
                        segment,
                        height=self.power_threshold * self.gap_power_factor,
                        prominence=self.peak_prominence * self.gap_prominence_factor
                    )
                    if pks.size > 0:
                        candidate_idxs = pks + lower_bin
                        candidate_freqs = f[candidate_idxs]
                        distances = np.abs(candidate_freqs - prev_freq)
                        valid_mask = distances <= self.gap_max_jump_hz
                        if np.any(valid_mask):
                            chosen_idx = np.argmin(distances * valid_mask + (~valid_mask) * 1e6)
                            matched_f_idx = candidate_idxs[chosen_idx]
                            new_list = track_list + [(ti, matched_f_idx)]
                            new_active.append((ti, matched_f_idx, 0, new_list))
                            continue

                # No match found: increment gap count
                new_active.append((last_ti, last_f_idx, gap_count + 1, track_list[:]))

            # 2. Start new tracks for unmatched peaks
            for pi, f_idx in enumerate(current_peaks):
                if pi not in used:
                    new_track = [(ti, f_idx)]
                    new_active.append((ti, f_idx, 0, new_track))

            # 3. Keep only tracks that have not exceeded gap limit
            active_tracks = []
            for last_ti, last_f_idx, gap_count, track_list in new_active:
                if gap_count > self.max_gap_frames:
                    finished_tracks.append(track_list)
                else:
                    active_tracks.append((last_ti, last_f_idx, gap_count, track_list))

        # 4. Finalize any remaining active tracks
        for last_ti, last_f_idx, gap_count, track_list in active_tracks:
            finished_tracks.append(track_list)

        # 5. Filter tracks by minimum length, average power, and frequency spread
        valid_tracks = []
        for track in finished_tracks:
            if len(track) < self.min_track_length_frames:
                continue
            f_idxs = [pt[1] for pt in track]
            powers = [Sxx_filtered[f_idx, ti] for ti, f_idx in track]
            avg_power = np.mean(powers)
            freqs_arr = f[f_idxs]
            freq_std = np.std(freqs_arr)

            if avg_power < self.min_track_avg_power:
                continue
            if freq_std > self.max_track_freq_std_hz:
                continue

            valid_tracks.append(track)

        return valid_tracks

    def merge_tracks(self, tracks):
        """
        Merge tracks that are close in time and frequency. Produces one combined track
        if two tracks’ start/end times are within merge_gap_frames AND their frequencies
        differ by no more than merge_max_freq_diff_hz.
        Returns a list of merged tracks.
        """
        f = self.freqs
        merged = []
        events = []
        for tr in tracks:
            tis = [pt[0] for pt in tr]
            f_idxs = [pt[1] for pt in tr]
            events.append({
                "track": tr,
                "start_ti": tis[0],
                "end_ti": tis[-1],
                "start_freq": f[f_idxs[0]],
                "end_freq": f[f_idxs[-1]]
            })

        # Sort events by start time
        events.sort(key=lambda e: e["start_ti"])
        for evt in events:
            if not merged:
                merged.append(evt.copy())
                continue

            last = merged[-1]
            time_gap = evt["start_ti"] - last["end_ti"]
            freq_gap = abs(evt["start_freq"] - last["end_freq"])
            if (time_gap <= self.merge_gap_frames) and (freq_gap <= self.merge_max_freq_diff_hz):
                # Combine the two tracks
                combined_track = last["track"] + evt["track"]
                last["track"] = combined_track
                last["end_ti"] = evt["end_ti"]
                last["end_freq"] = evt["end_freq"]
            else:
                merged.append(evt.copy())

        # Extract just the track lists
        merged_tracks = [evt["track"] for evt in merged]
        return merged_tracks

    def run_detection(self, filepath):
        """
        Full pipeline:
          1. Load audio
          2. Compute spectrogram + filters
          3. Detect peaks per frame
          4. Track peaks over time
          5. Merge similar tracks
          6. Final filtering of merged tracks by length, power, etc.

        Returns a list of final tracks. Each track is [(time_idx, freq_idx), ...].
        """
        y, sr = self.load_audio(filepath)
        f, t, Sxx_norm, Sxx_filt = self.compute_spectrogram(y, sr)

        peaks_per_frame = self.detect_peaks_per_frame()
        raw_tracks = self.track_peaks_over_time(peaks_per_frame)
        merged = self.merge_tracks(raw_tracks)

        # Final filter step
        final_tracks = []
        for tr in merged:
            if len(tr) < self.min_track_length_frames:
                continue
            f_idxs = [pt[1] for pt in tr]
            powers = [Sxx_filt[f_idx, ti] for ti, f_idx in tr]
            avg_power = np.mean(powers)
            freqs_arr = f[f_idxs]
            freq_std = np.std(freqs_arr)
            if avg_power < self.min_track_avg_power:
                continue
            if freq_std > self.max_track_freq_std_hz:
                continue
            final_tracks.append(tr)

        return final_tracks
