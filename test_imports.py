#!/usr/bin/env python3
"""
Test that all PySide6 imports work correctly after migration.
"""

import sys
import traceback

def test_imports():
    """Test all critical imports."""
    
    print("Testing PySide6 imports...")
    errors = []
    
    # Test basic PySide6 imports
    try:
        from PySide6.QtCore import Qt, Signal, QRunnable, QObject, QThreadPool, QSettings, QSize, QEvent
        print("✓ QtCore imports OK")
    except ImportError as e:
        errors.append(f"QtCore import error: {e}")
        print(f"✗ QtCore import failed: {e}")
    
    try:
        from PySide6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QFileDialog, QPushButton, QLabel, QListWidget, QListWidgetItem,
            QSplitter, QMessageBox, QMenu, QFrame,
            QToolButton, QGraphicsDropShadowEffect,
            QProgressDialog, QDialog, QTabWidget, QGroupBox, QFormLayout,
            QCheckBox, QComboBox, QSpinBox, QSlider, QDialogButtonBox
        )
        print("✓ QtWidgets imports OK")
    except ImportError as e:
        errors.append(f"QtWidgets import error: {e}")
        print(f"✗ QtWidgets import failed: {e}")
    
    try:
        from PySide6.QtGui import QKeySequence, QAction, QShortcut, QPalette, QColor, QFont
        print("✓ QtGui imports OK (including QAction and QShortcut)")
    except ImportError as e:
        errors.append(f"QtGui import error: {e}")
        print(f"✗ QtGui import failed: {e}")
    
    # Test our custom modules
    print("\nTesting application modules...")
    
    try:
        from spectrogram_gui.utils.worker_thread import WorkerSignals, Worker, ThreadPoolManager
        print("✓ worker_thread module OK")
    except ImportError as e:
        errors.append(f"worker_thread import error: {e}")
        print(f"✗ worker_thread import failed: {e}")
    
    try:
        from spectrogram_gui.gui.preferences_dialog import PreferencesDialog
        print("✓ preferences_dialog module OK")
    except ImportError as e:
        errors.append(f"preferences_dialog import error: {e}")
        print(f"✗ preferences_dialog import failed: {e}")
    
    try:
        from spectrogram_gui.gui.main_window import MainWindow
        print("✓ main_window module OK")
    except ImportError as e:
        errors.append(f"main_window import error: {e}")
        print(f"✗ main_window import failed: {e}")
    
    # Check for deprecated features
    print("\nChecking for deprecated features...")
    
    try:
        from PySide6.QtCore import Qt
        # This should not exist in Qt6
        if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
            print("⚠ Warning: AA_UseHighDpiPixmaps still accessible (deprecated)")
        else:
            print("✓ Deprecated AA_UseHighDpiPixmaps not used")
    except:
        pass
    
    # Summary
    print("\n" + "="*50)
    if errors:
        print(f"FAILED: {len(errors)} import errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("SUCCESS: All imports working correctly!")
        return True

if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)