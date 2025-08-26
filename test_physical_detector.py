#!/usr/bin/env python3
"""Test the Physical Doppler Detector with simulated data."""

import numpy as np
import sys
import os
sys.path.insert(0, '/workspace')

from spectrogram_gui.utils.physical_detector import PhysicalDopplerDetector, DopplerEvent
from scipy.signal import stft
import matplotlib.pyplot as plt

def test_detector_with_simulated_data():
    """Test the detector with simulated drone data."""
    
    print("Loading simulated drone data...")
    # Import the simulated data generation
    exec(open('/workspace/simulated.py').read(), globals())
    
    # We now have the simulated noisy_signals, fs, t variables
    print(f"Simulated data: {len(noisy_signals)} sensors, {len(t)} samples at {fs}Hz")
    
    # Test with first sensor signal
    signal = noisy_signals[0]
    
    # Create detector with parameters suitable for drones
    detector = PhysicalDopplerDetector(
        min_altitude_m=30.0,
        max_altitude_m=500.0,
        min_speed_ms=5.0,
        max_speed_ms=30.0,  # Drone speeds
        min_bpf_hz=100.0,   # Drone BPF range
        max_bpf_hz=400.0,
        bpf_tolerance_hz=20.0,
        min_event_duration_s=10.0,
        max_event_duration_s=120.0,
        min_snr_db=6.0,
        max_doppler_rate_hz_s=10.0,
        min_doppler_shift_hz=3.0
    )
    
    print("\nComputing spectrogram...")
    # Compute spectrogram using detector's method
    freqs, times_spec, Sxx = detector._compute_aircraft_spectrogram(signal, int(fs))
    print(f"Spectrogram shape: {Sxx.shape}, freq range: {freqs[0]:.1f}-{freqs[-1]:.1f}Hz")
    
    print("\nRunning detection...")
    events = detector.detect_events(Sxx, freqs, times_spec)
    
    print(f"\n=== Detection Results ===")
    print(f"Found {len(events)} events")
    
    for i, event in enumerate(events):
        print(f"\nEvent {i+1}:")
        print(f"  Time: {event.start_time:.1f}-{event.end_time:.1f}s (duration: {event.end_time - event.start_time:.1f}s)")
        print(f"  Type: {event.event_type}")
        print(f"  Confidence: {event.confidence:.2%}")
        print(f"  Frequency range: {event.frequency_range[0]:.1f}-{event.frequency_range[1]:.1f}Hz")
        print(f"  Closest approach: {event.closest_approach_time:.1f}s")
        print(f"  Number of tracks: {len(event.tracks)}")
        if event.doppler_signature:
            sig = event.doppler_signature
            print(f"  Doppler pattern: {sig.get('pattern', 'unknown')}")
            print(f"  Max Doppler rate: {sig.get('max_doppler_rate', 0):.2f} Hz/s")
            print(f"  Estimated speed: {sig.get('estimated_speed_ms', 0):.1f} m/s")
            print(f"  Estimated altitude: {sig.get('estimated_altitude_m', 0):.1f} m")
    
    # Visualize results
    if events:
        print("\nCreating visualization...")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot spectrogram
        ax1 = axes[0]
        im = ax1.pcolormesh(times_spec, freqs[:500], Sxx[:500, :], 
                           shading='gouraud', cmap='plasma')
        ax1.set_ylabel('Frequency [Hz]')
        ax1.set_title('Spectrogram with Detected Tracks')
        ax1.set_ylim(0, 500)
        
        # Overlay detected tracks
        colors = ['yellow', 'cyan', 'magenta', 'lime']
        for event_idx, event in enumerate(events):
            color = colors[event_idx % len(colors)]
            for track in event.tracks:
                if track:  # Check if track is not empty
                    track_times = [pt[0] for pt in track]
                    track_freqs = [pt[1] for pt in track]
                    ax1.plot(track_times, track_freqs, color=color, 
                            linewidth=2, alpha=0.8, 
                            label=f'Event {event_idx+1}' if track == event.tracks[0] else '')
        
        if events:
            ax1.legend(loc='upper right')
        
        # Plot amplitude envelope for first event
        if events[0].amplitude_envelope is not None and len(events[0].amplitude_envelope) > 0:
            ax2 = axes[1]
            event = events[0]
            # Create time array for the envelope
            env_times = np.linspace(event.start_time, event.end_time, len(event.amplitude_envelope))
            ax2.plot(env_times, event.amplitude_envelope, 'b-', linewidth=2)
            ax2.axvline(event.closest_approach_time, color='r', linestyle='--', 
                       label='Closest Approach')
            ax2.set_xlabel('Time [s]')
            ax2.set_ylabel('Amplitude')
            ax2.set_title(f'Event 1 Amplitude Envelope (Confidence: {event.confidence:.2%})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/workspace/detector_test_results.png', dpi=150)
        print(f"Visualization saved to detector_test_results.png")
        plt.close()
    
    return events

if __name__ == "__main__":
    events = test_detector_with_simulated_data()
    print(f"\n{'='*50}")
    print(f"Test completed. Detected {len(events)} events.")
    if events:
        print("✓ Detector appears to be working!")
    else:
        print("✗ No events detected - may need parameter tuning")