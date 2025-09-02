#!/usr/bin/env python3
"""
Modern Spectrogram GUI Application - Final Version
High-performance audio analysis with glassmorphic UI design
"""

import sys
import os
from pathlib import Path

# Ensure the parent directory is in the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QFontDatabase, QFont, QIcon

# Try to import qtawesome for icons
try:
    import qtawesome as qta
    ICONS_AVAILABLE = True
except ImportError:
    print("Warning: qtawesome not found. Icons will not be displayed.")
    print("Install with: pip install qtawesome")
    ICONS_AVAILABLE = False

from gui_modern.main_window_complete import CompleteModernMainWindow

# Application metadata
QCoreApplication.setOrganizationName("AudioAnalysis")
QCoreApplication.setApplicationName("SpectrogramAnalyzer")
QCoreApplication.setApplicationVersion("2.0.0")


def setup_application(app):
    """Configure application settings and style"""
    
    # Enable high DPI support
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    
    # Set application font
    font = QFont("Segoe UI", 10)
    if sys.platform == "darwin":  # macOS
        font = QFont("SF Pro Display", 10)
    elif sys.platform.startswith("linux"):
        font = QFont("Ubuntu", 10)
    
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app.setFont(font)
    
    # Set application icon if available
    if ICONS_AVAILABLE:
        app.setWindowIcon(qta.icon('fa5s.wave-square', color='#8B5CF6'))
    
    # Apply global stylesheet
    global_style = """
    /* Global font settings */
    * {
        font-family: 'Segoe UI', 'SF Pro Display', 'Ubuntu', 'Helvetica Neue', sans-serif;
    }
    
    /* Tooltips */
    QToolTip {
        background: rgba(20, 20, 30, 0.95);
        color: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 13px;
    }
    
    /* Scrollbars */
    QScrollBar:vertical {
        background: rgba(255, 255, 255, 0.02);
        width: 12px;
        border-radius: 6px;
        margin: 0;
    }
    
    QScrollBar::handle:vertical {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        min-height: 30px;
    }
    
    QScrollBar::handle:vertical:hover {
        background: rgba(255, 255, 255, 0.15);
    }
    
    QScrollBar::handle:vertical:pressed {
        background: rgba(139, 92, 246, 0.3);
    }
    
    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {
        height: 0;
        background: none;
    }
    
    QScrollBar:horizontal {
        background: rgba(255, 255, 255, 0.02);
        height: 12px;
        border-radius: 6px;
        margin: 0;
    }
    
    QScrollBar::handle:horizontal {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        min-width: 30px;
    }
    
    QScrollBar::handle:horizontal:hover {
        background: rgba(255, 255, 255, 0.15);
    }
    
    QScrollBar::handle:horizontal:pressed {
        background: rgba(139, 92, 246, 0.3);
    }
    
    QScrollBar::add-line:horizontal,
    QScrollBar::sub-line:horizontal {
        width: 0;
        background: none;
    }
    
    /* Menus */
    QMenu {
        background: rgba(20, 20, 30, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 4px;
    }
    
    QMenu::item {
        color: rgba(255, 255, 255, 0.9);
        padding: 8px 24px;
        border-radius: 4px;
        margin: 2px 4px;
    }
    
    QMenu::item:selected {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 rgba(99, 102, 241, 0.3),
            stop: 1 rgba(139, 92, 246, 0.3)
        );
    }
    
    QMenu::separator {
        height: 1px;
        background: rgba(255, 255, 255, 0.1);
        margin: 4px 8px;
    }
    
    /* Message boxes */
    QMessageBox {
        background: #1A1A2E;
        color: rgba(255, 255, 255, 0.9);
    }
    
    QMessageBox QPushButton {
        background: rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 8px 20px;
        min-width: 80px;
        font-weight: 500;
    }
    
    QMessageBox QPushButton:hover {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.5);
    }
    
    QMessageBox QPushButton:pressed {
        background: rgba(255, 255, 255, 0.03);
    }
    
    /* Dialogs */
    QDialog {
        background: #1A1A2E;
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Progress bars */
    QProgressBar {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        text-align: center;
        color: rgba(255, 255, 255, 0.9);
        height: 24px;
    }
    
    QProgressBar::chunk {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 #6366F1,
            stop: 1 #8B5CF6
        );
        border-radius: 7px;
    }
    
    /* Combo boxes */
    QComboBox {
        background: rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 6px 12px;
        min-width: 100px;
    }
    
    QComboBox:hover {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    
    QComboBox:focus {
        border: 1px solid rgba(139, 92, 246, 0.5);
        outline: none;
    }
    
    QComboBox::drop-down {
        border: none;
        width: 20px;
    }
    
    QComboBox::down-arrow {
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 6px solid rgba(255, 255, 255, 0.6);
        margin-right: 8px;
    }
    
    QComboBox QAbstractItemView {
        background: rgba(20, 20, 30, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        selection-background-color: rgba(139, 92, 246, 0.3);
        color: rgba(255, 255, 255, 0.9);
        padding: 4px;
    }
    
    /* Spin boxes */
    QSpinBox, QDoubleSpinBox {
        background: rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 6px 12px;
        min-width: 80px;
    }
    
    QSpinBox:hover, QDoubleSpinBox:hover {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    
    QSpinBox:focus, QDoubleSpinBox:focus {
        border: 1px solid rgba(139, 92, 246, 0.5);
        outline: none;
    }
    
    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {
        background: transparent;
        border: none;
        width: 16px;
    }
    
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
        border-left: 3px solid transparent;
        border-right: 3px solid transparent;
        border-bottom: 4px solid rgba(255, 255, 255, 0.6);
    }
    
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
        border-left: 3px solid transparent;
        border-right: 3px solid transparent;
        border-top: 4px solid rgba(255, 255, 255, 0.6);
    }
    
    /* Labels */
    QLabel {
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Group boxes */
    QGroupBox {
        color: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        margin-top: 8px;
        padding-top: 16px;
        font-weight: 500;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 8px;
    }
    
    /* Tab widgets */
    QTabWidget::pane {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    QTabBar::tab {
        background: rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.7);
        padding: 10px 20px;
        margin: 2px;
        border-radius: 6px;
    }
    
    QTabBar::tab:selected {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 0, y2: 1,
            stop: 0 rgba(139, 92, 246, 0.3),
            stop: 1 rgba(139, 92, 246, 0.2)
        );
        color: white;
    }
    
    QTabBar::tab:hover:!selected {
        background: rgba(255, 255, 255, 0.08);
    }
    """
    
    app.setStyleSheet(global_style)


def main():
    """Main application entry point"""
    # Create application instance
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationDisplayName("Spectrogram Analyzer - Modern UI")
    app.setApplicationName("SpectrogramAnalyzer")
    
    # Setup application style and configuration
    setup_application(app)
    
    # Create and show main window
    try:
        window = CompleteModernMainWindow()
        window.show()
        
        # Print startup message
        print("=" * 60)
        print("Spectrogram Analyzer - Modern UI v2.0.0")
        print("=" * 60)
        print("Features:")
        print("  • Glassmorphic UI design with smooth animations")
        print("  • High-performance spectrogram rendering")
        print("  • Smooth zoom with boundary constraints")
        print("  • Real-time audio playback with position tracking")
        print("  • Advanced filtering options")
        print("  • Optimized for FLAC and WAV files")
        print("=" * 60)
        print("Keyboard Shortcuts:")
        print("  • Space: Play/Pause")
        print("  • S: Stop")
        print("  • Ctrl+O: Open files")
        print("  • Ctrl+S: Export spectrogram")
        print("  • Ctrl++: Zoom in")
        print("  • Ctrl+-: Zoom out")
        print("  • Ctrl+0: Reset view")
        print("  • Delete: Remove selected files")
        print("  • Alt+hover: Show crosshair on spectrogram")
        print("  • Ctrl+click: Seek to position")
        print("=" * 60)
        print("Ready!")
        print()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Run the application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()