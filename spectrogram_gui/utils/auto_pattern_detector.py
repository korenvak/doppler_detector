import numpy as np
import time
from scipy import ndimage
from scipy.ndimage import binary_opening, binary_dilation
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
        # clip to avoid log of negative values
        Sxx_log = np.log1p(np.clip(self.Sxx_filt, a_min=0, a_max=None))
        Sxx_norm = (Sxx_log - Sxx_log.mean()) / (Sxx_log.std() + 1e-8)
        if getattr(self.detector, "use_median", False):
            Sxx_norm = ndimage.median_filter(Sxx_norm, size=3)
        self.Sxx_filt = Sxx_norm
        assert np.isfinite(self.Sxx_filt).all()
        print(
            f"[Debug] Preprocess: min={self.Sxx_filt.min():.3f}, max={self.Sxx_filt.max():.3f}"
        )

    def _ridge_mask(self, Sxx_band, sigma=2.5, threshold_percentile=85):
        """Enhanced ridge detection combining ridges and valleys."""
        # Normalize for better contrast
        Sxx_norm = (Sxx_band - np.mean(Sxx_band)) / (np.std(Sxx_band) + 1e-8)

        # Hessian matrix eigenvalues
        # explicit use_gaussian_derivatives to silence future warnings
        H = hessian_matrix(Sxx_norm, sigma=sigma, order="rc", use_gaussian_derivatives=False)
        ridges, valleys = hessian_matrix_eigvals(H)

        # Combine ridges and valleys
        combined = np.maximum(np.abs(ridges), np.abs(valleys))

        # Adaptive threshold
        if self.detector.adv_threshold_method == "otsu":
            from skimage.filters import threshold_otsu

            thr = threshold_otsu(combined)
            ridge_mask = combined > thr
        elif self.detector.adv_threshold_method == "cfar":
            ridge_mask = self._cfar_detection(
                combined,
                n_guard=self.detector.adv_cfar_guard,
                n_train=self.detector.adv_cfar_train,
                pfa=self.detector.adv_cfar_pfa,
            )
        else:
            if np.any(combined > 0):
                thr = np.percentile(
                    combined[combined > 0], threshold_percentile
                )
            else:
                thr = np.percentile(combined, threshold_percentile)
            ridge_mask = combined > thr
        return ridge_mask

    def _cfar_detection(self, Sxx_band, n_guard=2, n_train=10, pfa=0.001):
        """CFAR detection for robust track detection."""
        alpha = n_train * (pfa ** (-1 / n_train) - 1)
        detection_map = np.zeros_like(Sxx_band, dtype=bool)
        for j in range(Sxx_band.shape[1]):
            for i in range(n_guard + n_train, Sxx_band.shape[0] - n_guard - n_train):
                train_cells = np.concatenate(
                    [
                        Sxx_band[i - n_guard - n_train : i - n_guard, j],
                        Sxx_band[i + n_guard + 1 : i + n_guard + n_train + 1, j],
                    ]
                )
                if train_cells.size:
                    noise_level = np.mean(train_cells)
                    thr = alpha * noise_level
                    if Sxx_band[i, j] > thr:
                        detection_map[i, j] = True
        return detection_map

    def _calculate_track_confidence(self, band, track_points, offset):
        """Calculate confidence score for a track."""
        values = []
        for xi, yi in track_points:
            if 0 <= xi < band.shape[1] and 0 <= yi - offset < band.shape[0]:
                values.append(band[yi - offset, xi])
        if values:
            track_mean = np.mean(values)
            background_mean = np.mean(band)
            if background_mean > 0:
                return float(track_mean / background_mean)
            return 1.0
        return 0.5

    def _format_tracks(self, raw_tracks, band, freq_offset):
        """Format raw tracks into dicts with metadata."""
        formatted_tracks = []
        for pts in raw_tracks:
            if not pts:
                continue
            t_idx = np.array([p[0] for p in pts])
            f_local = np.array([p[1] for p in pts])
            f_idx = f_local + freq_offset
            valid_t = (t_idx >= 0) & (t_idx < len(self.times))
            valid_f = (f_idx >= 0) & (f_idx < len(self.freqs))
            valid = valid_t & valid_f
            if not np.any(valid):
                continue
            t_idx = t_idx[valid]
            f_idx = f_idx[valid]
            f_local = f_local[valid]
            duration = self.times[t_idx[-1]] - self.times[t_idx[0]]
            freq_change = self.freqs[f_idx[-1]] - self.freqs[f_idx[0]]
            if duration > 0:
                rate = freq_change / duration
            else:
                rate = 0.0
            info = {
                "times": self.times[t_idx],
                "freqs": self.freqs[f_idx],
                "indices": (t_idx, f_idx),
                "confidence": self._calculate_track_confidence(
                    band, list(zip(t_idx, f_local)), 0
                ),
                "method": "auto_pattern",
                "duration": float(duration),
                "freq_change": float(freq_change),
                "length": len(t_idx),
                "freq_change_rate": float(rate),
                "type": (
                    "stationary"
                    if abs(rate) < 10
                    else ("approaching" if rate > 0 else "receding")
                ),
            }
            formatted_tracks.append(info)
        formatted_tracks.sort(key=lambda x: x["confidence"], reverse=True)
        return formatted_tracks

    def detect_tracks_auto(self):
        """Detect patterns using ridge and CFAR masks with Hough lines."""
        start_t = time.perf_counter()

        # Preprocess spectrogram
        try:
            self._auto_preprocess()
        except Exception as e:
            print("[Error] preprocess", e)

        # Select frequency band
        f = self.freqs
        i_min = np.searchsorted(f, self.detector.freq_min, side="left")
        i_max = np.searchsorted(f, self.detector.freq_max, side="right") - 1
        i_min = max(0, i_min)
        i_max = min(len(f) - 1, i_max)
        Sxx = self.Sxx_filt
        band = Sxx[i_min : i_max + 1]

        # Create ridge & CFAR masks
        ridge_mask = self._ridge_mask(
            band,
            sigma=self.detector.adv_ridge_sigma,
            threshold_percentile=self.detector.adv_threshold_percentile,
        )
        cfar_mask = self._cfar_detection(
            band,
            n_guard=self.detector.adv_cfar_guard,
            n_train=self.detector.adv_cfar_train,
            pfa=self.detector.adv_cfar_pfa,
        )

        combined_mask = ridge_mask | cfar_mask

        mask = binary_opening(combined_mask, iterations=1)
        mask = binary_dilation(mask, iterations=1)
        mask = remove_small_objects(
            mask.astype(bool), min_size=self.detector.adv_min_object_size
        )
        if self.detector.adv_use_skeleton:
            mask = skeletonize(mask)

        # Hough transform
        if self.detector.deterministic_hough:
            h, theta, d = hough_line(mask.astype(np.uint8))
            acc, angles, dists = hough_line_peaks(h, theta, d, num_peaks=100)
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

        filtered_lines = []
        for (x0, y0), (x1, y1) in lines:
            dx = x1 - x0
            dy = y1 - y0
            length = np.sqrt(dx ** 2 + dy ** 2)
            if length < self.detector.adv_min_line_length:
                continue
            if hasattr(self.detector, "adv_min_slope") and self.detector.adv_min_slope > 0:
                if dx != 0:
                    slope = abs(dy / dx)
                    if slope >= self.detector.adv_min_slope or slope <= 0.1:
                        filtered_lines.append(((x0, y0), (x1, y1)))
                else:
                    filtered_lines.append(((x0, y0), (x1, y1)))
            else:
                filtered_lines.append(((x0, y0), (x1, y1)))

        raw_tracks = []
        for (x0, y0), (x1, y1) in filtered_lines:
            num = max(abs(x1 - x0), abs(y1 - y0)) + 1
            xs = np.linspace(x0, x1, num).astype(int)
            ys = np.linspace(y0, y1, num).astype(int)
            track_points = []
            for xi, yi in zip(xs, ys):
                if 0 <= xi < mask.shape[1] and 0 <= yi < mask.shape[0]:
                    track_points.append((xi, yi))
            if len(track_points) >= self.detector.adv_min_line_length:
                raw_tracks.append(track_points)

        formatted_tracks = self._format_tracks(raw_tracks, band, i_min)

        print(
            f"[AutoPatternDetector] Detected {len(formatted_tracks)} tracks in {time.perf_counter() - start_t:.2f}s"
        )
        return formatted_tracks

