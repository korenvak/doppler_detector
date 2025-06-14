import numpy as np
import time
from scipy import ndimage
from scipy.ndimage import binary_opening
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.transform import probabilistic_hough_line, hough_line, hough_line_peaks
from skimage.morphology import remove_small_objects, skeletonize


class AutoPatternDetector:
    """Helper that detects track-like patterns on an existing spectrogram."""

    def __init__(self, detector):
        self.detector = detector
        self.freqs = detector.freqs
        self.times = detector.times
        self.Sxx_filt = detector.Sxx_filt

    def _auto_preprocess(self):
        """Adaptive preprocessing with optional median filter."""
        assert isinstance(self.Sxx_filt, np.ndarray) and self.Sxx_filt.ndim == 2
        assert np.isfinite(self.Sxx_filt).all()
        Sxx_log = np.log1p(self.Sxx_filt)
        Sxx_norm = (Sxx_log - Sxx_log.mean()) / (Sxx_log.std() + 1e-8)
        if getattr(self.detector, "use_median", False):
            Sxx_norm = ndimage.median_filter(Sxx_norm, size=3)
        self.Sxx_filt = Sxx_norm
        assert np.isfinite(self.Sxx_filt).all()
        print(
            f"[Debug] Preprocess: min={self.Sxx_filt.min():.3f}, max={self.Sxx_filt.max():.3f}"
        )

    def _ridge_mask(self, Sxx_band, sigma=2.5):
        H = hessian_matrix(
            Sxx_band, sigma=sigma, order="rc", use_gaussian_derivatives=False
        )
        ridges, _ = hessian_matrix_eigvals(H)
        mag = np.abs(ridges)
        if self.detector.adv_threshold_method == "otsu":
            from skimage.filters import threshold_otsu

            thr = threshold_otsu(mag)
            ridge_mask = mag > thr
        elif self.detector.adv_threshold_method == "cfar":
            ridge_mask = self.detector._cfar(
                mag,
                num_train=self.detector.adv_cfar_train,
                num_guard=self.detector.adv_cfar_guard,
                pfa=self.detector.adv_cfar_pfa,
            )
        else:
            thr = np.percentile(mag, self.detector.adv_threshold_percentile)
            ridge_mask = mag > thr
        return ridge_mask

    def detect_tracks_auto(self):
        """Run full preprocessing, dynamic threshold ridge detection, morphology, deterministic Hough, and post-merge filtering. Vectorized for speed, with feedback loop for threshold auto-tuning."""
        start_t = time.perf_counter()
        try:
            self._auto_preprocess()
        except Exception as e:
            print("[Error] preprocess", e)

        try:
            f = self.freqs
            i_min = np.searchsorted(f, self.detector.freq_min, side="left")
            i_max = np.searchsorted(f, self.detector.freq_max, side="right") - 1
            i_min = max(0, i_min)
            i_max = min(len(f) - 1, i_max)
            band = self.Sxx_filt[i_min : i_max + 1]

            ridge_mask = self._ridge_mask(
                band, sigma=self.detector.adv_ridge_sigma
            )
            coverage = ridge_mask.mean()
            print(f"[Debug] mask coverage={coverage*100:.1f}%")
            if coverage > self.detector.max_mask_coverage:
                self.detector.adv_threshold_percentile += (
                    self.detector.threshold_adjust_step
                )
            elif coverage < self.detector.min_mask_coverage:
                self.detector.adv_threshold_percentile -= (
                    self.detector.threshold_adjust_step
                )

            mask = binary_opening(ridge_mask, iterations=1)
            mask = remove_small_objects(
                mask.astype(bool), min_size=self.detector.adv_min_object_size
            )
            if self.detector.adv_use_skeleton:
                mask = skeletonize(mask)
        except Exception as e:
            print("[Error] ridge", e)
            return []

        try:
            if self.detector.deterministic_hough:
                h, theta, d = hough_line(mask.astype(np.uint8))
                acc, angles, dists = hough_line_peaks(
                    h, theta, d, num_peaks=100
                )
                lines = []
                for angle, dist in zip(angles, dists):
                    x0 = 0
                    y0 = (dist - x0 * np.cos(angle)) / np.sin(angle)
                    x1 = mask.shape[1]
                    y1 = (dist - x1 * np.cos(angle)) / np.sin(angle)
                    lines.append(((x0, y0), (x1, y1)))
            else:
                np.random.seed(self.detector.random_seed)
                lines = probabilistic_hough_line(
                    mask.astype(np.uint8),
                    threshold=10,
                    line_length=self.detector.adv_min_line_length,
                    line_gap=self.detector.adv_line_gap,
                )
        except Exception as e:
            print("[Error] hough", e)
            lines = []

        filtered_lines = []
        for (x0, y0), (x1, y1) in lines:
            dx = x1 - x0
            dy = y1 - y0
            if dx == 0:
                continue
            slope = dy / dx
            if abs(slope) >= self.detector.adv_min_slope:
                filtered_lines.append(((x0, y0), (x1, y1)))
        print(f"[Debug] raw lines={len(filtered_lines)}")

        raw_tracks = []
        width = mask.shape[1]
        height = mask.shape[0]
        for (x0, y0), (x1, y1) in filtered_lines:
            length = max(abs(x1 - x0), abs(y1 - y0)) + 1
            xs = np.linspace(x0, x1, length, dtype=int)
            ys = np.linspace(y0, y1, length, dtype=int) + i_min
            valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height + i_min)
            xs, ys = xs[valid], ys[valid]
            if xs.size:
                raw_tracks.append({
                    "indices": (xs, ys),
                    "confidence": 1.0,
                    "method": "pattern",
                })

        track_lists = [list(zip(t["indices"][0], t["indices"][1])) for t in raw_tracks]
        merged = self.detector.merge_tracks(track_lists)

        final_tracks = []
        for tr in merged:
            if len(tr) < self.detector.min_track_length_frames:
                continue
            f_idxs = [pt[1] for pt in tr]
            powers = [self.Sxx_filt[fi, ti] for ti, fi in tr]
            if np.mean(powers) < self.detector.min_track_avg_power:
                continue
            if np.std(self.freqs[f_idxs]) > self.detector.max_track_freq_std_hz:
                continue
            t_idx = np.array([pt[0] for pt in tr], dtype=int)
            f_idx = np.array(f_idxs, dtype=int)
            duration = self.times[t_idx[-1]] - self.times[t_idx[0]]
            freq_change = self.freqs[f_idx[-1]] - self.freqs[f_idx[0]]
            rate = freq_change / (duration + 1e-9)
            final_tracks.append(
                {
                    "indices": (t_idx, f_idx),
                    "confidence": 1.0,
                    "method": "pattern",
                    "duration": float(duration),
                    "freq_change": float(freq_change),
                    "freq_change_rate": float(rate),
                    "type": "up" if rate >= 0 else "down",
                }
            )

        print(
            f"[AutoPatternDetector] Detected {len(final_tracks)} tracks in {time.perf_counter() - start_t:.2f}s"
        )
        return final_tracks

