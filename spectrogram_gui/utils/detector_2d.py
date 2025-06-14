import numpy as np
from skimage.feature import peak_local_max

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
        max_peaks_per_frame=30,
        min_track_length_frames=10,
        min_track_avg_power=0.08,
        merge_gap_frames=150,
        merge_max_freq_diff_hz=40.0,
        power_threshold=0.2,
        peak_prominence=0.185,
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

        coords = peak_local_max(
            S,
            footprint=np.ones((3, 3)),
            threshold_abs=self.power_threshold,
            threshold_rel=self.peak_prominence,
            exclude_border=1,
        )
        if coords.size == 0:
            return [[] for _ in range(S.shape[1])], [[] for _ in range(S.shape[1])]

        fr_idx = coords[:, 0]
        t_idx = coords[:, 1]

        mask = (
            (self.freqs[fr_idx] >= self.freq_min)
            & (self.freqs[fr_idx] <= self.freq_max)
        )
        above_idx = np.clip(fr_idx - 1, 0, S.shape[0] - 1)
        below_idx = np.clip(fr_idx + 1, 0, S.shape[0] - 1)
        prominence = S[fr_idx, t_idx] - np.maximum(S[above_idx, t_idx], S[below_idx, t_idx])
        mask &= prominence >= self.peak_prominence

        fr_idx = fr_idx[mask]
        t_idx = t_idx[mask]
        if fr_idx.size == 0:
            return [[] for _ in range(S.shape[1])], [[] for _ in range(S.shape[1])]

        conf = S[fr_idx, t_idx]

        order = np.lexsort((-conf, t_idx))
        fr_idx = fr_idx[order]
        t_idx = t_idx[order]
        conf = conf[order]

        peaks_per_frame = [[] for _ in range(S.shape[1])]
        conf_per_frame = [[] for _ in range(S.shape[1])]

        last_t = -1
        count = 0
        for f, t, c in zip(fr_idx, t_idx, conf):
            if t != last_t:
                last_t = t
                count = 0
            if count < self.max_peaks_per_frame:
                peaks_per_frame[t].append(f)
                conf_per_frame[t].append(c)
                count += 1

        return peaks_per_frame, conf_per_frame

    # ----- Tracking -----
    def predict_next_position(self, last_f, prev_f, last_t, prev_t, gap):
        vel = (self.freqs[last_f] - self.freqs[prev_f]) / max(last_t - prev_t, 1)
        return self.freqs[last_f] + vel * gap

    def _predict_positions(self, last_f, prev_f, last_t, prev_t, gaps):
        last_f = np.asarray(last_f)
        prev_f = np.asarray(prev_f)
        last_t = np.asarray(last_t)
        prev_t = np.asarray(prev_t)
        gaps = np.asarray(gaps)
        vel = (self.freqs[last_f] - self.freqs[prev_f]) / np.maximum(last_t - prev_t, 1)
        return self.freqs[last_f] + vel * gaps

    def track_peaks_enhanced(self, peaks_per_frame, conf_per_frame, progress_callback=None):
        finished = []
        active = []  # (last_t, last_f, prev_t, prev_f, gap, trace, conf)

        for ti, (peaks, confs) in enumerate(zip(peaks_per_frame, conf_per_frame)):
            peaks_arr = np.asarray(peaks, dtype=np.int64)
            confs_arr = np.asarray(confs, dtype=np.float64)
            used = np.zeros(len(peaks_arr), dtype=bool)
            new_active = []

            if active and len(peaks_arr):
                last_t = np.array([a[0] for a in active])
                last_f = np.array([a[1] for a in active])
                prev_t = np.array([a[2] for a in active])
                prev_f = np.array([a[3] for a in active])
                gaps = np.array([a[4] for a in active])
                traces = [a[5] for a in active]
                conf_tr = np.array([a[6] for a in active])

                pred = self._predict_positions(last_f, prev_f, last_t, prev_t, gaps + 1)
                peak_freqs = self.freqs[peaks_arr]
                diff = np.abs(peak_freqs[None, :] - pred[:, None])
                allowed = diff <= self.max_freq_jump_hz * (1 + gaps[:, None] * 0.2)
                scores = np.where(allowed, confs_arr[None, :] / (1 + diff / 10), -np.inf)
                best_idx = np.argmax(scores, axis=1)
                best_score = scores[np.arange(scores.shape[0]), best_idx]
                order = np.argsort(-best_score)

                for ai in order:
                    sc = best_score[ai]
                    if sc == -np.inf:
                        new_active.append((last_t[ai], last_f[ai], prev_t[ai], prev_f[ai], gaps[ai] + 1, traces[ai], conf_tr[ai] * 0.9))
                        continue
                    pk = best_idx[ai]
                    if used[pk]:
                        new_active.append((last_t[ai], last_f[ai], prev_t[ai], prev_f[ai], gaps[ai] + 1, traces[ai], conf_tr[ai] * 0.9))
                        continue
                    used[pk] = True
                    tr = traces[ai] + [(ti, peaks_arr[pk])]
                    new_active.append((ti, peaks_arr[pk], last_t[ai], last_f[ai], 0, tr, conf_tr[ai] * 0.9 + confs_arr[pk] * 0.1))

            for idx, f_idx in enumerate(peaks_arr):
                if not used[idx]:
                    new_active.append((ti, f_idx, ti, f_idx, 0, [(ti, f_idx)], confs_arr[idx]))

            still_active = []
            for a in new_active:
                if a[4] > self.max_gap_frames:
                    finished.append(a[5])
                else:
                    still_active.append(a)

            active = sorted(still_active, key=lambda x: x[6], reverse=True)[:100]

            if progress_callback and ti % 10 == 0:
                progress_callback(ti)

        for a in active:
            finished.append(a[5])
        if progress_callback:
            progress_callback(len(peaks_per_frame))
        return finished

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
