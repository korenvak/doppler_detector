import numpy as np
from multiprocessing import Pool
from collections import defaultdict

from .core import DopplerDetector

def tracks_are_similar(tr1, tr2, overlap_thresh=0.6):
    """
    Check if two tracks overlap in at least overlap_thresh fraction
    of their points. Each track is a list of (time_idx, freq_idx).
    """
    set1 = set(tr1)
    set2 = set(tr2)
    if not set1 or not set2:
        return False
    intersection = set1 & set2
    overlap = len(intersection) / min(len(set1), len(set2))
    return overlap >= overlap_thresh

class VotingDetector:
    def __init__(self, base_detector: DopplerDetector, param_grid: list, num_processes: int = None):
        """
        base_detector: an instance of DopplerDetector (with default params).
        param_grid: a list of dicts, each dict containing one configuration.
        num_processes: number of CPU cores to use; if None, use all available.
        """
        self.base = base_detector
        self.param_grid = param_grid
        self.N_RUNS = len(param_grid)

        # Determine number of processes
        if num_processes is None or num_processes < 1:
            from multiprocessing import cpu_count
            self.num_processes = cpu_count()
        else:
            self.num_processes = num_processes

    def _worker(self, args):
        index, params, f, t, Sxx_filtered = args

        # Temporarily update base_detector’s parameters
        self.base.power_threshold = params["POWER_THRESHOLD"]
        self.base.peak_prominence = params["PEAK_PROMINENCE"]
        self.base.max_freq_jump_hz = params["MAX_FREQ_JUMP_HZ"]
        self.base.min_track_length_frames = params["MIN_TRACK_LENGTH_FRAMES"]
        self.base.max_track_freq_std_hz = params["MAX_TRACK_FREQ_STD_HZ"]
        self.base.gap_prominence_factor = params["GAP_PROMINENCE_FACTOR"]
        self.base.gap_power_factor = params["GAP_POWER_FACTOR"]

        peaks = self.base.detect_peaks_per_frame()
        tracks = self.base.track_peaks_over_time(peaks)
        merged = self.base.merge_tracks(tracks)

        result = []
        for tr in merged:
            unique_key = tuple((ti, fi) for ti, fi in tr)
            result.append((unique_key, tr, index))
        return result

    def run_voting(self, filepath, progress_dialog=None):
        """
        1. Load audio and compute spectrogram once (shared by all configs).
        2. For each config, run in parallel (num_processes) to collect tracks.
        3. Build occurrence_map and vote_strength_map, compute metrics for each track.
        4. Pre-filter based on scores, then merge similar pre-filtered tracks.
        Returns:
            f: frequency bins
            t: time bins
            Sxx_norm: normalized spectrogram
            final_tracks: list of merged tracks after voting
        """
        y, sr = self.base.load_audio(filepath)
        f, t, Sxx_norm, Sxx_filtered = self.base.compute_spectrogram(y, sr)

        args_list = []
        for i, params in enumerate(self.param_grid):
            args_list.append((i, params, f, t, Sxx_filtered))

        occurrence_map = defaultdict(int)
        vote_strength_map = defaultdict(list)
        track_lookup = []

        # If a progress_dialog was passed, configure its range:
        if progress_dialog is not None:
            progress_dialog.setMaximum(self.N_RUNS)
            progress_dialog.setValue(0)
            progress_dialog.setLabelText(f"Processed 0 / {self.N_RUNS}")

        processed = 0
        with Pool(processes=self.num_processes) as pool:
            for config_results in pool.imap_unordered(self._worker, args_list):
                processed += 1
                if progress_dialog is not None:
                    progress_dialog.setValue(processed)
                    progress_dialog.setLabelText(f"Processed {processed} / {self.N_RUNS}")
                for unique_key, tr, idx in config_results:
                    track_lookup.append((unique_key, tr, idx))
                    for ti, fi in tr:
                        occurrence_map[(ti, fi)] += 1
                        vote_strength_map[(ti, fi)].append(idx)

        # Pre-filter tracks based on combined metrics
        MIN_RATIO = 0.3
        track_lengths = [p["MIN_TRACK_LENGTH_FRAMES"] for p in self.param_grid]
        pre_filtered_tracks = []
        seen = set()

        for key, tr, config_index in track_lookup:
            if key in seen:
                continue
            seen.add(key)

            votes = [len(vote_strength_map[(ti, fi)]) for ti, fi in tr]
            vote_ratio = np.mean(votes) / self.N_RUNS

            freqs_arr = np.array([f[fi] for _, fi in tr])
            jump_score = np.mean(np.abs(np.diff(freqs_arr)) < 40)
            length_score = len(tr) / max(track_lengths)
            std_score = max(0, 1 - np.std(freqs_arr) / 70)

            # Compute a simple harmonic score
            HARMONIC_THRESHOLD = 0.05
            MAX_HARMONIC_MULT = 5
            harmonic_count = 0
            for ti, fi in tr:
                base_freq = f[fi]
                harm_freq = base_freq * MAX_HARMONIC_MULT
                harm_idx = np.searchsorted(f, harm_freq)
                if harm_idx < len(f):
                    if Sxx_filtered[harm_idx, ti] >= HARMONIC_THRESHOLD:
                        harmonic_count += 1
            harmonic_score = harmonic_count / len(tr)

            combined_score = (
                0.3 * length_score
                + 0.3 * std_score
                + 0.15 * vote_ratio
                + 0.25 * harmonic_score
            )

            if (
                vote_ratio >= MIN_RATIO
                and len(tr) >= min(track_lengths)
                and np.std(freqs_arr) < 70
                and jump_score > 0.8
                and combined_score > 0.45
            ):
                pre_filtered_tracks.append(tr)

        # Merge similar pre-filtered tracks
        final_tracks = []
        used = set()
        for i, tr1 in enumerate(pre_filtered_tracks):
            if i in used:
                continue
            group = [tr1]
            for j in range(i + 1, len(pre_filtered_tracks)):
                if j in used:
                    continue
                tr2 = pre_filtered_tracks[j]
                if tracks_are_similar(tr1, tr2):
                    group.append(tr2)
                    used.add(j)
            used.add(i)

            # Combine group tracks by averaging freq per time index
            merged = []
            time_bins = defaultdict(list)
            for tr in group:
                for ti, fi in tr:
                    time_bins[ti].append(f[fi])
            for ti in sorted(time_bins):
                avg_freq = np.mean(time_bins[ti])
                closest_fi = np.argmin(np.abs(f - avg_freq))
                merged.append((ti, closest_fi))
            final_tracks.append(merged)

        return f, t, Sxx_norm, final_tracks
