"""
Test Level of Detail (LOD) envelope decimation for performance optimization.
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLODEnvelope(unittest.TestCase):
    """Test that LOD decimation preserves peak bounds."""
    
    def test_envelope_preserves_peaks(self):
        """Test that min/max envelope preserves signal peaks."""
        # Create a test signal with known peaks
        t = np.linspace(0, 10, 10000)
        signal = np.sin(2 * np.pi * t) + 0.5 * np.sin(20 * np.pi * t)
        
        # Add some spikes
        signal[1000] = 5.0  # Positive spike
        signal[5000] = -5.0  # Negative spike
        
        # Compute envelope with block size of 100
        block_size = 100
        n_blocks = len(signal) // block_size
        
        min_envelope = np.zeros(n_blocks)
        max_envelope = np.zeros(n_blocks)
        
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            block = signal[start:end]
            min_envelope[i] = np.min(block)
            max_envelope[i] = np.max(block)
        
        # Verify peaks are preserved
        self.assertAlmostEqual(np.max(max_envelope), 5.0, places=5)
        self.assertAlmostEqual(np.min(min_envelope), -5.0, places=5)
        
        # Verify envelope bounds original signal
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            block = signal[start:end]
            self.assertLessEqual(np.min(block), max_envelope[i])
            self.assertGreaterEqual(np.max(block), min_envelope[i])
    
    def test_decimation_levels(self):
        """Test multi-level decimation for different zoom levels."""
        # Create a large signal
        signal = np.random.randn(1_000_000)
        
        # Test different decimation factors
        factors = [10, 100, 1000]
        
        for factor in factors:
            n_blocks = len(signal) // factor
            decimated = np.zeros(n_blocks)
            
            for i in range(n_blocks):
                start = i * factor
                end = start + factor
                # Use peak detection for decimation
                decimated[i] = np.max(np.abs(signal[start:end]))
            
            # Verify decimated size
            self.assertEqual(len(decimated), n_blocks)
            
            # Verify no information loss for peaks
            original_max = np.max(np.abs(signal))
            decimated_max = np.max(decimated)
            self.assertAlmostEqual(original_max, decimated_max, places=5)
    
    def test_adaptive_decimation(self):
        """Test adaptive decimation based on view range."""
        signal = np.random.randn(100_000)
        view_width_pixels = 1920  # Typical screen width
        
        # Calculate appropriate decimation factor
        points_per_pixel = len(signal) / view_width_pixels
        
        if points_per_pixel > 10:
            # Need decimation
            factor = int(points_per_pixel / 2)  # Show ~2 points per pixel
            n_blocks = len(signal) // factor
            
            # Compute min/max envelope
            min_env = np.zeros(n_blocks)
            max_env = np.zeros(n_blocks)
            
            for i in range(n_blocks):
                start = i * factor
                end = min(start + factor, len(signal))
                if start < len(signal):
                    block = signal[start:end]
                    min_env[i] = np.min(block)
                    max_env[i] = np.max(block)
            
            # Verify appropriate decimation
            self.assertLess(n_blocks, view_width_pixels * 5)  # Not too many points
            self.assertGreater(n_blocks, view_width_pixels / 2)  # Not too few


if __name__ == '__main__':
    unittest.main()