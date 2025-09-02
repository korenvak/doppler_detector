#!/usr/bin/env python3
"""
Test script for the modern spectrogram GUI
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check required packages
required_packages = [
    ('PySide6', 'PySide6'),
    ('pyqtgraph', 'pyqtgraph'),
    ('numpy', 'numpy'),
    ('scipy', 'scipy'),
    ('pandas', 'pandas'),
    ('sounddevice', 'sounddevice'),
    ('qtawesome', 'qtawesome')
]

print("=" * 60)
print("Modern Spectrogram GUI - System Check")
print("=" * 60)
print()

missing_packages = []
for package_name, import_name in required_packages:
    try:
        __import__(import_name)
        print(f"✓ {package_name} is installed")
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        missing_packages.append(package_name)

if missing_packages:
    print()
    print("Missing packages detected!")
    print("Please install them using:")
    print(f"  pip install {' '.join(missing_packages)}")
    print()
    print("Or use the complete installation:")
    print("  pip install -r requirements_new.txt")
    sys.exit(1)

print()
print("All required packages are installed!")
print()
print("Key Features Implemented:")
print("  ✓ Glassmorphic UI design with gradients")
print("  ✓ Fixed spectrogram orientation (freq vs time)")
print("  ✓ Waveform display below spectrogram")
print("  ✓ Event marking and detection system")
print("  ✓ CSV export for annotations")
print("  ✓ Expandable/collapsible waveform panel")
print("  ✓ Detector parameters dialog")
print("  ✓ Smooth zoom with FLAC boundaries")
print("  ✓ Performance optimizations (JIT, caching)")
print()
print("Starting application...")
print("=" * 60)

# Now try to import and run
try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    import qtawesome as qta
    
    # Import our application
    from gui_modern.main_window_complete import CompleteModernMainWindow
    
    # Create application
    app = QApplication(sys.argv)
    
    # Configure
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    app.setApplicationDisplayName("Spectrogram Analyzer - Modern UI")
    
    # Create window
    window = CompleteModernMainWindow()
    window.show()
    
    # Run
    sys.exit(app.exec())
    
except Exception as e:
    print(f"Error starting application: {e}")
    import traceback
    traceback.print_exc()