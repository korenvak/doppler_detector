import numpy as np
import time
from scipy.ndimage import binary_opening
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.transform import probabilistic_hough_line
from skimage.morphology import remove_small_objects, skeletonize


class AutoPatternDetector:
    """Helper that detects track-like patterns on an existing spectrogram."""

    def __init__(self, detector):
        self.detector = detector
        self.freqs = detector.freqs
        self.times = detector.times
        self.Sxx_filt = detector.Sxx_filt

    def _ridge_mask(self, Sxx_band, sigma=2.5, threshold_percentile=85):
        H = hessian_matrix(Sxx_band, sigma=sigma, order='rc', use_gaussian_derivatives=False)
        ridges, _ = hessian_matrix_eigvals(H)
        thr = np.percentile(np.abs(ridges), threshold_percentile)
        return np.abs(ridges) > thr

    def detect_tracks_auto(self):
        start_t = time.perf_counter()

        f = self.freqs
        Sxx = np.log1p(self.Sxx_filt)

        i_min = np.searchsorted(f, self.detector.freq_min, side='left')
        i_max = np.searchsorted(f, self.detector.freq_max, side='right') - 1
        i_min = max(0, i_min)
        i_max = min(len(f) - 1, i_max)
        band = Sxx[i_min:i_max + 1]

        ridge_mask = self._ridge_mask(
            band,
            sigma=self.detector.adv_ridge_sigma,
            threshold_percentile=self.detector.adv_threshold_percentile,
        )

        mask = binary_opening(ridge_mask, iterations=1)
        mask = remove_small_objects(mask.astype(bool), min_size=self.detector.adv_min_object_size)

        if self.detector.adv_use_skeleton:
            mask = skeletonize(mask)

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
            if dx == 0:
                continue
            slope = dy / dx
            if abs(slope) >= self.detector.adv_min_slope:
                filtered_lines.append(((x0, y0), (x1, y1)))

        tracks = []
        for (x0, y0), (x1, y1) in filtered_lines:
            num = max(abs(x1 - x0), abs(y1 - y0)) + 1
            xs = np.linspace(x0, x1, num).astype(int)
            ys = np.linspace(y0, y1, num).astype(int)
            tr = []
            for xi, yi in zip(xs, ys):
                if 0 <= xi < mask.shape[1] and 0 <= yi < mask.shape[0]:
                    tr.append((xi, yi + i_min))
            if tr:
                tracks.append(tr)

        print(f"[AutoPatternDetector] Detected {len(tracks)} tracks in {time.perf_counter() - start_t:.2f}s")
        return tracks
