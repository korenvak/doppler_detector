import numpy as np
import time
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, median_filter, label, laplace, binary_opening
from scipy.signal import find_peaks, wiener
from skimage.transform import probabilistic_hough_line
from skimage.morphology import remove_small_objects, skeletonize
# use your spectrogram and audio‐loading utils instead of soundfile + scipy.spectrogram
from spectrogram_gui.utils.audio_utils import load_audio_with_filters
from spectrogram_gui.utils.spectrogram_utils import compute_spectrogram as sg_compute_spec
from spectrogram_gui.utils.filter_utils import (
    apply_nlms,
    apply_ale_2d_doppler_wave,
    apply_wiener_adaptive,
)

# Default frequency range (Hz)
DEFAULT_FREQ_MIN = 50

DEFAULT_FREQ_MAX = 1500


class Detector:
    """Simple detector base class"""

    def load_audio(self, filepath):
        raise NotImplementedError

    def compute_spectrogram(self, y, sr, filepath):
        raise NotImplementedError

    def run_detection(self, filepath):
        raise NotImplementedError


class DopplerDetector(Detector):
    def __init__(
        self,
        freq_min=DEFAULT_FREQ_MIN,
        freq_max=DEFAULT_FREQ_MAX,
        power_threshold=0.2,
        peak_prominence=0.185,
        max_gap_frames=4,
        gap_power_factor=0.8,
        gap_prominence_factor=0.8,
        max_freq_jump_hz=15.0,
        gap_max_jump_hz=10.0,
        max_peaks_per_frame=20,
        min_track_length_frames=14,
        min_track_avg_power=0.1,
        max_track_freq_std_hz=70.0,
        merge_gap_frames=100,
        merge_max_freq_diff_hz=30.0,
        smooth_sigma=1.5,
        median_filter_size=(3, 1),
        detection_method="peaks",
        adv_threshold_percentile=80,
        adv_min_line_length=30,
        adv_line_gap=10,
        adv_use_cfar=True,
        adv_ridge_sigma=2.5,
        adv_cfar_train=20,
        adv_cfar_guard=2,
        adv_cfar_pfa=0.001,
        adv_min_object_size=30,
        adv_use_skeleton=True,
        adv_min_slope=0.3,
        adv_threshold_method="percentile",
        random_seed=0,
        deterministic_hough=False,
        use_median=False,
        max_mask_coverage=0.25,
        min_mask_coverage=0.05,
        threshold_adjust_step=1.0,
        enable_post_merge=True,
        fast_mode=False,
    ):
        # detection parameters
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
        self.detection_method = detection_method
        self.adv_threshold_percentile = adv_threshold_percentile
        self.adv_min_line_length = adv_min_line_length
        self.adv_line_gap = adv_line_gap
        self.adv_use_cfar = adv_use_cfar
        self.adv_ridge_sigma = adv_ridge_sigma
        self.adv_cfar_train = adv_cfar_train
        self.adv_cfar_guard = adv_cfar_guard
        self.adv_cfar_pfa = adv_cfar_pfa
        self.adv_min_object_size = adv_min_object_size
        self.adv_use_skeleton = adv_use_skeleton
        self.adv_min_slope = adv_min_slope
        self.adv_threshold_method = adv_threshold_method
        self.random_seed = random_seed
        self.deterministic_hough = deterministic_hough
        self.use_median = use_median
        self.max_mask_coverage = max_mask_coverage
        self.min_mask_coverage = min_mask_coverage
        self.threshold_adjust_step = threshold_adjust_step
        self.enable_post_merge = enable_post_merge
        self.fast_mode = fast_mode

        # will be set in compute_spectrogram
        self.freqs = None
        self.times = None
        self.Sxx_filt = None

        # spectrogram GUI params (must be injected by caller)
        self.spectrogram_params = {}

    def load_audio(self, filepath):
        """
        Load and filter audio via spectrogram_gui utilities.
        """
        return load_audio_with_filters(filepath)

    def compute_spectrogram(self, y, sr, filepath):
        """
        Delegate spectrogram to spectrogram_gui's compute_spectrogram.
        """
        freqs, times, Sxx_norm, Sxx_filt = sg_compute_spec(
            y, sr, filepath, params=self.spectrogram_params
        )
        self.freqs = freqs
        self.times = times
        self.Sxx_filt = Sxx_filt
        return freqs, times, Sxx_norm, Sxx_filt

    def detect_peaks_per_frame(self):
        """
        Find peaks in each time‐bin of the filtered spectrogram.
        """
        f = self.freqs
        S = self.Sxx_filt
        # frequency index bounds
        i_min = np.searchsorted(f, self.freq_min, side="left")
        i_max = np.searchsorted(f, self.freq_max, side="right") - 1
        i_min = max(i_min, 0)
        i_max = min(i_max, len(f) - 1)

        num_t = S.shape[1]
        peaks_per_frame = [[] for _ in range(num_t)]
        for ti in tqdm(range(num_t), desc="Detecting peaks per frame"):
            col = S[i_min : i_max + 1, ti]
            idxs, props = find_peaks(
                col,
                height=self.power_threshold,
                prominence=self.peak_prominence
            )
            if idxs.size:
                heights = props["peak_heights"]
                order = np.argsort(heights)[::-1][: self.max_peaks_per_frame]
                absolute = (idxs[order] + i_min).tolist()
            else:
                absolute = []
            peaks_per_frame[ti] = absolute
        return peaks_per_frame

    def track_peaks_over_time(self, peaks_per_frame):
        """
        Link peaks frame‐to‐frame into continuous tracks.
        """
        f = self.freqs
        S = self.Sxx_filt
        finished = []
        active = []  # tuples: (last_ti, last_fi, gap, track)

        for ti, frame_peaks in enumerate(tqdm(peaks_per_frame, desc="Tracking peaks")):
            used = set()
            new_active = []

            # extend existing
            for last_ti, last_fi, gap, track in active:
                if gap > self.max_gap_frames:
                    finished.append(track)
                    continue

                prev_freq = f[last_fi]
                prev_power = S[last_fi, last_ti]
                match = None
                best_d = None

                # strict match
                for pi, fi in enumerate(frame_peaks):
                    if pi in used: continue
                    cf = f[fi]
                    d = abs(cf - prev_freq)
                    if d <= self.max_freq_jump_hz and S[fi, ti] >= prev_power * self.gap_power_factor:
                        if best_d is None or d < best_d:
                            best_d, match = d, pi

                if match is not None:
                    used.add(match)
                    fi = frame_peaks[match]
                    new_active.append((ti, fi, 0, track + [(ti, fi)]))
                    continue

                # gap match
                lower = prev_freq - self.gap_max_jump_hz
                upper = prev_freq + self.gap_max_jump_hz
                li = np.searchsorted(f, lower, side="left")
                ui = np.searchsorted(f, upper, side="right") - 1
                li, ui = max(li, 0), min(ui, len(f) - 1)
                segment = S[li:ui+1, ti]
                pks, _ = find_peaks(
                    segment,
                    height=self.power_threshold * self.gap_power_factor,
                    prominence=self.peak_prominence * self.gap_prominence_factor
                )
                if pks.size:
                    cand = pks + li
                    dists = np.abs(f[cand] - prev_freq)
                    best = cand[np.argmin(dists)]
                    new_active.append((ti, best, 0, track + [(ti, best)]))
                else:
                    new_active.append((last_ti, last_fi, gap+1, track))

            # start new
            for pi, fi in enumerate(frame_peaks):
                if pi not in used:
                    new_active.append((ti, fi, 0, [(ti, fi)]))

            # prune
            active = [t for t in new_active if t[2] <= self.max_gap_frames]
            finished.extend([t[3] for t in new_active if t[2] > self.max_gap_frames])

        # finish leftovers
        finished.extend([t[3] for t in active])

        # filter out short / weak tracks
        valid = []
        for tr in finished:
            if len(tr) < self.min_track_length_frames:
                continue
            f_idxs = [pt[1] for pt in tr]
            powers = [S[fi, ti] for ti, fi in tr]
            if np.mean(powers) < self.min_track_avg_power:
                continue
            if np.std(f[f_idxs]) > self.max_track_freq_std_hz:
                continue
            valid.append(tr)
        return valid

    def merge_tracks(self, tracks):
        """
        Merge nearby tracks in time & frequency.
        """
        f = self.freqs
        events = []
        for tr in tracks:
            tis = [pt[0] for pt in tr]
            fis = [pt[1] for pt in tr]
            events.append({
                "track": tr,
                "start": tis[0], "end": tis[-1],
                "fstart": f[fis[0]], "fend": f[fis[-1]]
            })
        events.sort(key=lambda e: e["start"])
        merged = []
        for e in events:
            if not merged:
                merged.append(e)
            else:
                last = merged[-1]
                if e["start"] - last["end"] <= self.merge_gap_frames and abs(e["fstart"]-last["fend"]) <= self.merge_max_freq_diff_hz:
                    last["track"].extend(e["track"])
                    last["end"] = e["end"]
                    last["fend"] = e["fend"]
                else:
                    merged.append(e)
        return [e["track"] for e in merged]

    def _preprocess_spectrogram(self, Sxx):
        S = np.log1p(Sxx)
        S = wiener(S, (5, 5))
        S = S - 0.3 * laplace(S)
        return S

    def _cfar(self, Sxx, num_train=20, num_guard=2, pfa=0.001):
        """Return a boolean mask using a simple cell-averaging CFAR."""
        n_freq, _ = Sxx.shape
        alpha = num_train * (pfa ** (-1 / num_train) - 1)
        mask = np.zeros_like(Sxx, dtype=bool)
        for fi in range(num_guard + num_train, n_freq - num_guard - num_train):
            up = Sxx[fi - num_guard - num_train : fi - num_guard]
            down = Sxx[fi + num_guard + 1 : fi + num_guard + num_train + 1]
            if up.size and down.size:
                noise = 0.5 * (up.mean(axis=0) + down.mean(axis=0))
                thr = alpha * noise
                mask[fi] = Sxx[fi] > thr
        return mask


    def detect_tracks_advanced(self):
        start_t = time.perf_counter()
        S = self._preprocess_spectrogram(self.Sxx_filt)
        i_min = np.searchsorted(self.freqs, self.freq_min, side='left')
        i_max = np.searchsorted(self.freqs, self.freq_max, side='right') - 1
        i_min = max(i_min, 0)
        i_max = min(i_max, len(self.freqs) - 1)
        band = S[i_min:i_max + 1]
        thr = np.percentile(band, self.adv_threshold_percentile)
        base_m = band > thr
        if self.adv_use_cfar:
            cfar_m = self._cfar(
                band,
                num_train=self.adv_cfar_train,
                num_guard=self.adv_cfar_guard,
                pfa=self.adv_cfar_pfa,
            )
        else:
            cfar_m = np.ones_like(base_m, dtype=bool)

        mask = base_m | cfar_m
        print(
            "Base:", base_m.sum(),
            "CFAR:", cfar_m.sum(),
            "Combined:", mask.sum(),
        )
        mask = binary_opening(mask, iterations=1)
        mask = remove_small_objects(mask.astype(bool), min_size=self.adv_min_object_size)
        if self.adv_use_skeleton:
            mask = skeletonize(mask)
        lines = probabilistic_hough_line(
            mask.astype(np.uint8),
            threshold=10,
            line_length=self.adv_min_line_length,
            line_gap=self.adv_line_gap,
        )
        filtered_lines = []
        for (x0, y0), (x1, y1) in lines:
            dx = x1 - x0
            dy = y1 - y0
            if dx == 0:
                continue
            slope = dy / dx
            if abs(slope) >= self.adv_min_slope:
                filtered_lines.append(((x0, y0), (x1, y1)))

        tracks = []
        for line in filtered_lines:
            (x0, y0), (x1, y1) = line
            num = max(abs(x1 - x0), abs(y1 - y0)) + 1
            xs = np.linspace(x0, x1, num).astype(int)
            ys = np.linspace(y0, y1, num).astype(int)
            tr = []
            for xi, yi in zip(xs, ys):
                if 0 <= xi < mask.shape[1] and 0 <= yi < mask.shape[0]:
                    tr.append((xi, yi + i_min))
            if tr:
                tracks.append(tr)
        print(f"[Pattern] detection {time.perf_counter()-start_t:.2f}s")
        return tracks


    def run_detection(self, filepath):
        """
        Full pipeline: load, spec, detect, track, merge, filter.
        Returns list of tracks (each a list of (time_idx, freq_idx)).
        """
        start_t = time.perf_counter()
        y, sr = self.load_audio(filepath)
        f, t, Sxx_norm, Sxx_filt = self.compute_spectrogram(y, sr, filepath)
        if self.detection_method == "advanced":
            tracks = self.detect_tracks_advanced()
        else:
            peaks = self.detect_peaks_per_frame()
            tracks = self.track_peaks_over_time(peaks)

        tracks = self.merge_tracks(tracks)

        final = []
        for tr in tracks:
            if len(tr) < self.min_track_length_frames:
                continue
            f_idxs = [pt[1] for pt in tr]
            powers = [Sxx_filt[fi, ti] for ti, fi in tr]
            if np.mean(powers) < self.min_track_avg_power:
                continue
            if np.std(f[f_idxs]) > self.max_track_freq_std_hz:
                continue
            final.append(tr)

        print(f"[Detect] full pipeline {time.perf_counter()-start_t:.2f}s")
        return final


class AdaptiveFilterDetector(DopplerDetector):
    """Doppler detector with extra adaptive filtering"""

    def __init__(
        self,
        *args,
        nlms_mu=0.01,
        ale_delay=None,
        ale_mu=0.1,
        ale_lambda=0.995,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.nlms_mu = nlms_mu
        self.ale_delay = ale_delay
        self.ale_mu = ale_mu
        self.ale_lambda = ale_lambda

    def load_audio(self, filepath):
        y, sr = super().load_audio(filepath)
        order = min(32, len(y))
        if order > 1:
            y = apply_nlms(y, mu=self.nlms_mu, filter_order=order)
        if self.ale_delay is None or order > self.ale_delay:
            y = apply_ale_2d_doppler_wave(
                y,
                delay=self.ale_delay if self.ale_delay is not None else 3,
                mu=self.ale_mu,
                filter_order=order,
            )
        y = apply_wiener_adaptive(y, window_size=1024)
        return y, sr
