"""
Level of Detail (LOD) decimator for efficient rendering of large datasets.
Preserves peaks and visual features while reducing point count.
"""

import numpy as np
from typing import Tuple, Optional
from doppler_detector.spectrogram_gui.utils.logger import debug, timer


class LODDecimator:
    """
    Efficient decimation of time series data for rendering.
    Uses min/max envelope to preserve peaks at any zoom level.
    """
    
    def __init__(self, threshold: int = 50000, max_points: int = 10000):
        """
        Initialize the decimator.
        
        Args:
            threshold: Number of points above which decimation is applied
            max_points: Target number of points after decimation
        """
        self.threshold = threshold
        self.max_points = max_points
        self._cache = {}
        
    def clear_cache(self):
        """Clear the decimation cache."""
        self._cache.clear()
        debug("LOD cache cleared")
    
    def decimate(self, x: np.ndarray, y: np.ndarray, 
                 view_range: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decimate data for efficient rendering.
        
        Args:
            x: X-axis data (time)
            y: Y-axis data (amplitude)
            view_range: Optional (xmin, xmax) to limit decimation to visible range
            
        Returns:
            Decimated (x, y) arrays suitable for plotting
        """
        # Extract view range if specified
        if view_range is not None:
            xmin, xmax = view_range
            mask = (x >= xmin) & (x <= xmax)
            x_view = x[mask]
            y_view = y[mask]
        else:
            x_view = x
            y_view = y
        
        n_points = len(x_view)
        
        # No decimation needed for small datasets
        if n_points <= self.threshold:
            debug(f"No decimation needed: {n_points} points")
            return x_view, y_view
        
        # Calculate decimation factor
        factor = max(1, n_points // self.max_points)
        
        # Check cache
        cache_key = (id(x), id(y), factor, view_range)
        if cache_key in self._cache:
            debug(f"Using cached decimation: factor={factor}")
            return self._cache[cache_key]
        
        with timer(f"Decimating {n_points} points by factor {factor}"):
            # Compute min/max envelope
            x_dec, y_dec = self._compute_envelope(x_view, y_view, factor)
            
            # Cache result
            self._cache[cache_key] = (x_dec, y_dec)
            
            # Limit cache size
            if len(self._cache) > 10:
                # Remove oldest entries
                keys = list(self._cache.keys())
                for k in keys[:5]:
                    del self._cache[k]
            
            debug(f"Decimated {n_points} → {len(x_dec)} points")
            return x_dec, y_dec
    
    def _compute_envelope(self, x: np.ndarray, y: np.ndarray, 
                         factor: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute min/max envelope for decimation.
        
        This creates a visual envelope that preserves all peaks and valleys,
        ensuring no visual information is lost during decimation.
        """
        n_points = len(x)
        n_blocks = n_points // factor
        
        # Pre-allocate arrays for envelope
        # Each block contributes 2 points (min and max)
        x_env = np.zeros(n_blocks * 2)
        y_env = np.zeros(n_blocks * 2)
        
        for i in range(n_blocks):
            start = i * factor
            end = min(start + factor, n_points)
            
            # Get block data
            x_block = x[start:end]
            y_block = y[start:end]
            
            if len(y_block) > 0:
                # Find min and max in block
                min_idx = np.argmin(y_block)
                max_idx = np.argmax(y_block)
                
                # Store min/max points (ordered by x position)
                if min_idx < max_idx:
                    x_env[i*2] = x_block[min_idx]
                    y_env[i*2] = y_block[min_idx]
                    x_env[i*2 + 1] = x_block[max_idx]
                    y_env[i*2 + 1] = y_block[max_idx]
                else:
                    x_env[i*2] = x_block[max_idx]
                    y_env[i*2] = y_block[max_idx]
                    x_env[i*2 + 1] = x_block[min_idx]
                    y_env[i*2 + 1] = y_block[min_idx]
        
        # Remove any zeros from incomplete blocks
        valid = x_env != 0
        return x_env[valid], y_env[valid]


class AdaptiveLOD:
    """
    Adaptive LOD that adjusts decimation based on view width and data density.
    """
    
    def __init__(self, target_ppd: float = 2.0):
        """
        Initialize adaptive LOD.
        
        Args:
            target_ppd: Target points per display pixel
        """
        self.target_ppd = target_ppd
        self.decimator = LODDecimator()
        
    def process(self, x: np.ndarray, y: np.ndarray,
                view_range: Tuple[float, float],
                display_width: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process data with adaptive decimation based on display parameters.
        
        Args:
            x: X-axis data
            y: Y-axis data  
            view_range: (xmin, xmax) of visible range
            display_width: Width of display area in pixels
            
        Returns:
            Appropriately decimated (x, y) for current view
        """
        xmin, xmax = view_range
        
        # Get data in view range
        mask = (x >= xmin) & (x <= xmax)
        x_view = x[mask]
        y_view = y[mask]
        
        n_points = len(x_view)
        
        # Calculate points per pixel
        if display_width > 0:
            points_per_pixel = n_points / display_width
        else:
            points_per_pixel = 0
            
        debug(f"Adaptive LOD: {n_points} points, {points_per_pixel:.1f} pts/pixel")
        
        # Determine if decimation is needed
        if points_per_pixel > self.target_ppd * 2:
            # Too many points, need decimation
            target_points = int(display_width * self.target_ppd)
            self.decimator.max_points = target_points
            return self.decimator.decimate(x, y, view_range)
        else:
            # No decimation needed
            return x_view, y_view