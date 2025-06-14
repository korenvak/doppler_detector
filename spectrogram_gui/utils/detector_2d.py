import numpy as np
from scipy.ndimage import maximum_filter


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
        """Fast 2D local maxima detection using a small maximum filter."""
        S = self.Sxx_filt
        freq_mask = (self.freqs >= self.freq_min) & (self.freqs <= self.freq_max)
        sub = S[freq_mask]
        local_max = maximum_filter(sub, size=3, mode="constant") == sub
        mask = (sub >= self.power_threshold) & local_max
        fr_idx, t_idx = np.nonzero(mask)
        global_f = np.nonzero(freq_mask)[0][fr_idx]

        num_t = S.shape[1]
        peaks_per_frame = [[] for _ in range(num_t)]
        conf_per_frame = [[] for _ in range(num_t)]
        for f_i, t_i in zip(global_f, t_idx):
            peaks_per_frame[t_i].append(f_i)
            conf_per_frame[t_i].append(S[f_i, t_i])

        for ti in range(num_t):
            if len(peaks_per_frame[ti]) > self.max_peaks_per_frame:
                order = np.argsort(conf_per_frame[ti])[::-1][: self.max_peaks_per_frame]
                peaks_per_frame[ti] = [peaks_per_frame[ti][i] for i in order]
                conf_per_frame[ti] = [conf_per_frame[ti][i] for i in order]
        return peaks_per_frame, conf_per_frame

    # ----- Tracking -----
    def predict_next_position(self, track, gap):
        if len(track) < 3:
            if len(track) >= 2:
                t0, f0 = track[-1]
                t1, f1 = track[-2]
                vel = (self.freqs[f0] - self.freqs[f1]) / (t0 - t1)
                return self.freqs[f0] + vel * gap
            return self.freqs[track[-1][1]]
        tis = np.array([pt[0] for pt in track[-5:]])
        fis = np.array([self.freqs[pt[1]] for pt in track[-5:]])
        coeffs = np.polyfit(tis, fis, 2)
        next_t = track[-1][0] + gap
        return np.polyval(coeffs, next_t)

    def track_peaks_enhanced(self, peaks_per_frame, conf_per_frame, progress_callback=None):
        finished = []
        active = []
        for ti, (peaks, confs) in enumerate(zip(peaks_per_frame, conf_per_frame)):
            used = set()
            new_active = []
            for last_t, last_f, gap, tr, tr_conf in active:
                if gap > self.max_gap_frames:
                    finished.append(tr)
                    continue
                pred_freq = self.predict_next_position(tr, gap + 1)
                best = None
                best_score = -np.inf
                for idx, f_idx in enumerate(peaks):
                    if idx in used:
                        continue
                    freq_val = self.freqs[f_idx]
                    if abs(freq_val - pred_freq) <= self.max_freq_jump_hz * (1 + gap * 0.2):
                        score = confs[idx] / (1 + abs(freq_val - pred_freq) / 10)
                        if score > best_score:
                            best_score = score
                            best = idx
                if best is not None:
                    used.add(best)
                    new_tr = tr + [(ti, peaks[best])]
                    new_conf = tr_conf * 0.9 + confs[best] * 0.1
                    new_active.append((ti, peaks[best], 0, new_tr, new_conf))
                else:
                    new_active.append((last_t, last_f, gap + 1, tr, tr_conf * 0.9))
            for idx, f_idx in enumerate(peaks):
                if idx not in used:
                    new_active.append((ti, f_idx, 0, [(ti, f_idx)], confs[idx]))
            active = sorted(
                [a for a in new_active if a[2] <= self.max_gap_frames],
                key=lambda x: x[4],
                reverse=True,
            )[:100]
            if progress_callback and ti % 10 == 0:
                progress_callback(ti)
        for a in active:
            finished.append(a[3])
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
