#!/usr/bin/env python3
"""Simple test of the Physical Doppler Detector."""

import numpy as np
import sys
sys.path.insert(0, '/workspace')

from spectrogram_gui.utils.physical_detector import PhysicalDopplerDetector

def test_detector_simple():
    """Test the detector with synthetic data."""
    
    print("Creating synthetic test data...")
    
    # Create a simple synthetic Doppler signal
    fs = 8000  # Sample rate
    duration = 30  # seconds
    t = np.linspace(0, duration, int(fs * duration))
    
    # Create a simple Doppler shift pattern
    # Frequency that increases then decreases (flyover pattern)
    f_center = 200  # Hz - center frequency
    f_shift = 20  # Hz - Doppler shift amount
    
    # Create Doppler curve - approach then recede
    doppler_curve = f_shift * np.cos(2 * np.pi * t / duration)
    f_inst = f_center + doppler_curve
    
    # Generate signal with amplitude envelope
    amplitude = np.exp(-((t - duration/2) / (duration/4))**2)  # Gaussian envelope
    signal = amplitude * np.sin(2 * np.pi * np.cumsum(f_inst) / fs)
    
    # Add some noise
    signal += 0.01 * np.random.randn(len(signal))
    
    print(f"Generated {len(signal)} samples at {fs}Hz")
    
    # Create detector
    detector = PhysicalDopplerDetector(
        min_altitude_m=30.0,
        max_altitude_m=500.0,
        min_speed_ms=5.0,
        max_speed_ms=30.0,
        min_bpf_hz=150.0,
        max_bpf_hz=250.0,
        bpf_tolerance_hz=20.0,
        min_event_duration_s=5.0,
        max_event_duration_s=60.0,
        min_snr_db=3.0,
        max_doppler_rate_hz_s=15.0,
        min_doppler_shift_hz=5.0
    )
    
    print("\nComputing spectrogram...")
    freqs, times, Sxx = detector._compute_aircraft_spectrogram(signal, fs)
    print(f"Spectrogram shape: {Sxx.shape}")
    
    print("\nRunning detection...")
    events = detector.detect_events(Sxx, freqs, times)
    
    print(f"\n=== Detection Results ===")
    print(f"Found {len(events)} events")
    
    if events:
        for i, event in enumerate(events):
            print(f"\nEvent {i+1}:")
            print(f"  Time: {event.start_time:.1f}-{event.end_time:.1f}s")
            print(f"  Type: {event.event_type}")
            print(f"  Confidence: {event.confidence:.2%}")
            print(f"  Frequency range: {event.frequency_range[0]:.1f}-{event.frequency_range[1]:.1f}Hz")
            print(f"  Number of tracks: {len(event.tracks)}")
        print("\n✓ Detector is working!")
    else:
        print("\n⚠ No events detected - detector may need parameter tuning")
    
    return len(events) > 0

if __name__ == "__main__":
    success = test_detector_simple()
    sys.exit(0 if success else 1)