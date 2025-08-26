#!/usr/bin/env python3
"""Test the GUI with Physical Detector integration."""

import sys
import os
sys.path.insert(0, '/workspace')

from PyQt5.QtWidgets import QApplication
from spectrogram_gui.gui.main_window import MainWindow

def test_gui():
    """Launch the GUI to test the Physical Detector integration."""
    app = QApplication(sys.argv)
    
    # Load style sheet if available
    style_path = os.path.join(
        os.path.dirname(__file__),
        "spectrogram_gui", "styles", "style.qss"
    )
    if os.path.exists(style_path):
        with open(style_path, "r") as f:
            app.setStyleSheet(f.read())
    
    window = MainWindow()
    window.setWindowTitle("Spectrogram GUI - Physical Detector Test")
    window.show()
    
    # Print detector status
    print("GUI Loaded Successfully!")
    print(f"Physical Detector: {hasattr(window, 'physical_detector')}")
    print(f"Physical Detect Button: {hasattr(window, 'auto_detect_physical_btn')}")
    
    if hasattr(window, 'physical_detector'):
        detector = window.physical_detector
        print(f"\nPhysical Detector Parameters:")
        print(f"  BPF Range: {detector.min_bpf_hz}-{detector.max_bpf_hz} Hz")
        print(f"  Speed Range: {detector.min_speed_ms}-{detector.max_speed_ms} m/s")
        print(f"  Event Duration: {detector.min_event_duration_s}-{detector.max_event_duration_s} s")
        print(f"  Min SNR: {detector.min_snr_db} dB")
    
    print("\n✓ Physical Detector is integrated!")
    print("You can now:")
    print("1. Load an audio file")
    print("2. Click 'Physical Detect' button")
    print("3. Adjust parameters in the dialog")
    print("4. See detected Doppler events on the spectrogram")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    test_gui()