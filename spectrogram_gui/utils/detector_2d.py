import numpy as np
from scipy.ndimage import maximum_filter

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - numba may not be installed
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        def wrapper(func):
            return func

        return wrapper


class DopplerDetector2D:
    """Doppler track detector using 2D peak picking"""

    def __init__(
        self,
        freq_min=50,
        freq_max=1500,
        max_gap_frames=6,
        gap_power_factor=0.7,
        gap_prominence_factor=0.7,
        max_freq_jump_hz=20.0,
        gap_max_jump_hz=15.0,
        max_peaks_per_frame=10,
        min_track_length_frames=10,
        min_track_avg_power=0.08,
        max_track_freq_std_hz=70.0,
        merge_gap_frames=150,
        merge_max_freq_diff_hz=40.0,
        power_threshold=0.25,
        peak_prominence=0.25,
    ):
        self.freq_min = freq_min
        self.freq_max = freq_max
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
        self.power_threshold = power_threshold
        self.peak_prominence = peak_prominence
        self.freqs = None
        self.times = None
        self.Sxx_filt = None

    # ----- Peak detection -----
    def detect_peaks_2d(self):
        """Find local maxima in the entire spectrogram."""
        S = self.Sxx_filt

        # work only on the allowed frequency band
        freq_mask = (self.freqs >= self.freq_min) & (self.freqs <= self.freq_max)
        sub = S[freq_mask]

        # local maxima using a small maximum filter
        local_max = maximum_filter(sub, size=3, mode="nearest")
        mask = (sub == local_max) & (sub >= self.power_threshold)
        fr_sub, t_idx = np.nonzero(mask)
        if fr_sub.size == 0:
            return [[] for _ in range(S.shape[1])], [[] for _ in range(S.shape[1])]

        fr_idx = np.nonzero(freq_mask)[0][fr_sub]

        above_idx = np.clip(fr_idx - 1, 0, S.shape[0] - 1)
        below_idx = np.clip(fr_idx + 1, 0, S.shape[0] - 1)
        prominence = S[fr_idx, t_idx] - np.maximum(S[above_idx, t_idx], S[below_idx, t_idx])
        mask = prominence >= self.peak_prominence

        fr_idx = fr_idx[mask]
        t_idx = t_idx[mask]
        if fr_idx.size == 0:
            return [[] for _ in range(S.shape[1])], [[] for _ in range(S.shape[1])]

        conf = S[fr_idx, t_idx]

        order = np.argsort(t_idx)
        fr_idx = fr_idx[order]
        t_idx = t_idx[order]
        conf = conf[order]

        peaks_per_frame = [[] for _ in range(S.shape[1])]
        conf_per_frame = [[] for _ in range(S.shape[1])]

        i = 0
        n = len(t_idx)
        while i < n:
            t = t_idx[i]
            j = i
            while j < n and t_idx[j] == t:
                j += 1

            c_slice = conf[i:j]
            f_slice = fr_idx[i:j]
            if c_slice.size > self.max_peaks_per_frame:
                part = np.argpartition(-c_slice, self.max_peaks_per_frame - 1)[: self.max_peaks_per_frame]
                part_order = part[np.argsort(-c_slice[part])]
                peaks_per_frame[t] = f_slice[part_order].tolist()
                conf_per_frame[t] = c_slice[part_order].tolist()
            else:
                order2 = np.argsort(-c_slice)
                peaks_per_frame[t] = f_slice[order2].tolist()
                conf_per_frame[t] = c_slice[order2].tolist()
            i = j

        return peaks_per_frame, conf_per_frame

    # ----- Tracking -----
    def track_peaks_enhanced(self, peaks_per_frame, conf_per_frame, progress_callback=None):
        finished = []
        last_t = np.empty(0, dtype=np.int64)
        last_f = np.empty(0, dtype=np.int64)
        gaps = np.empty(0, dtype=np.int64)
        traces = []

        for ti, (peaks, confs) in enumerate(zip(peaks_per_frame, conf_per_frame)):
            peaks_arr = np.asarray(peaks, dtype=np.int64)
            confs_arr = np.asarray(confs, dtype=np.float64)
            used = np.zeros(len(peaks_arr), dtype=bool)

            new_last_t = []
            new_last_f = []
            new_gaps = []
            new_traces = []

            for li in range(len(last_t)):
                prev_freq = self.freqs[last_f[li]]
                diff = np.abs(self.freqs[peaks_arr] - prev_freq)
                mask = diff <= self.max_freq_jump_hz
                if np.any(mask):
                    cand = np.where(mask)[0]
                    idx = cand[np.argmax(confs_arr[cand])]
                    used[idx] = True
                    fi = peaks_arr[idx]
                    tr = traces[li] + [(ti, fi)]
                    new_last_t.append(ti)
                    new_last_f.append(fi)
                    new_gaps.append(0)
                    new_traces.append(tr)
                else:
                    gap = gaps[li] + 1
                    if gap > self.max_gap_frames:
                        finished.append(traces[li])
                    else:
                        new_last_t.append(last_t[li])
                        new_last_f.append(last_f[li])
                        new_gaps.append(gap)
                        new_traces.append(traces[li])

            for idx, fi in enumerate(peaks_arr):
                if not used[idx]:
                    new_last_t.append(ti)
                    new_last_f.append(fi)
                    new_gaps.append(0)
                    new_traces.append([(ti, fi)])

            last_t = np.array(new_last_t, dtype=np.int64)
            last_f = np.array(new_last_f, dtype=np.int64)
            gaps = np.array(new_gaps, dtype=np.int64)
            traces = new_traces

            keep = gaps <= self.max_gap_frames
            if not np.all(keep):
                finished.extend([traces[i] for i, k in enumerate(keep) if not k])
                last_t = last_t[keep]
                last_f = last_f[keep]
                gaps = gaps[keep]
                traces = [traces[i] for i, k in enumerate(keep) if k]

            if progress_callback and ti % 10 == 0:
                progress_callback(ti)

        finished.extend(traces)
        if progress_callback:
            progress_callback(len(peaks_per_frame))

        valid = []
        S = self.Sxx_filt
        f = self.freqs
        for tr in finished:
            if len(tr) < self.min_track_length_frames:
                continue
            powers = np.array([S[fi, ti] for ti, fi in tr])
            if powers.mean() < self.min_track_avg_power:
                continue
            if np.std(f[[pt[1] for pt in tr]]) > self.max_track_freq_std_hz:
                continue
            valid.append(tr)
        return valid

    # ----- Filtering -----
    def filter_and_score_tracks(self, tracks):
        scored = []
        S = self.Sxx_filt
        f = self.freqs
        for tr in tracks:
            if len(tr) < self.min_track_length_frames:
                continue
            tis = np.array([pt[0] for pt in tr])
            fis = np.array([pt[1] for pt in tr])
            if np.std(self.freqs[fis]) > self.max_track_freq_std_hz:
                continue
            powers = np.array([S[f_i, t_i] for t_i, f_i in tr])
            avg_pow = powers.mean()
            if avg_pow < self.min_track_avg_power:
                continue
            score = 0
            score += min(len(tr) / 50, 1.0) * 0.2
            score += (1 / (1 + powers.std() / (avg_pow + 1e-8))) * 0.3
            if len(tr) > 2:
                diffs = np.diff(f[fis])
                score += (1 / (1 + diffs.std() / 10)) * 0.3
            else:
                score += 0.15
            noise = []
            for ti, fi in tr:
                for off in (-5, 5):
                    fi2 = fi + off
                    if 0 <= fi2 < len(f):
                        noise.append(S[fi2, ti])
            snr = avg_pow / (np.median(noise) + 1e-8) if noise else 1
            score += min(snr / 10, 1.0) * 0.2
            if score > 0:
                scored.append(tr)
        return scored

    # ----- Merging -----
    def merge_tracks_advanced(self, tracks):
        if not tracks:
            return []
        events = []
        for tr in tracks:
            tis = [pt[0] for pt in tr]
            fis = [pt[1] for pt in tr]
            events.append({
                "track": tr,
                "start": tis[0],
                "end": tis[-1],
                "fstart": self.freqs[fis[0]],
                "fend": self.freqs[fis[-1]],
            })
        events.sort(key=lambda e: e["start"])
        merged = []
        for e in events:
            if not merged:
                merged.append(e)
                continue
            last = merged[-1]
            if (
                e["start"] - last["end"] <= self.merge_gap_frames
                and abs(e["fstart"] - last["fend"]) <= self.merge_max_freq_diff_hz
            ):
                last["track"].extend(e["track"])
                last["end"] = e["end"]
                last["fend"] = e["fend"]
            else:
                merged.append(e)
        return [e["track"] for e in merged]

    # ----- Pipeline -----
    def run_detection(self, Sxx, freqs, times, progress_callback=None):
        self.Sxx_filt = Sxx
        self.freqs = freqs
        self.times = times
        peaks, confs = self.detect_peaks_2d()
        raw = self.track_peaks_enhanced(peaks, confs, progress_callback=progress_callback)
        scored = self.filter_and_score_tracks(raw)
        final = self.merge_tracks_advanced(scored)
        return final
