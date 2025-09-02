#!/usr/bin/env python3
"""
Modern Spectrogram GUI Application
High-performance audio analysis with glassmorphic UI design
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QFontDatabase, QFont

from gui_modern.main_window import ModernMainWindow

# Application metadata
QCoreApplication.setOrganizationName("AudioAnalysis")
QCoreApplication.setApplicationName("SpectrogramAnalyzer")
QCoreApplication.setApplicationVersion("2.0.0")


def load_fonts():
    """Load custom fonts for the application"""
    # Try to load system fonts or fallback to defaults
    font_db = QFontDatabase()
    
    # Set application font
    app_font = QFont("Segoe UI", 10)
    app_font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    QApplication.setFont(app_font)


def apply_global_style(app):
    """Apply global application style"""
    # Enable high DPI support
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    
    # Set global stylesheet
    global_style = """
    * {
        font-family: 'Segoe UI', 'SF Pro Display', 'Helvetica Neue', sans-serif;
    }
    
    QToolTip {
        background: rgba(20, 20, 30, 0.95);
        color: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 8px;
        padding: 8px;
        font-size: 13px;
    }
    
    QMessageBox {
        background: #1A1A2E;
        color: rgba(255, 255, 255, 0.9);
    }
    
    QMessageBox QPushButton {
        background: rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 8px 16px;
        min-width: 80px;
    }
    
    QMessageBox QPushButton:hover {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.5);
    }
    
    QDialog {
        background: #1A1A2E;
        color: rgba(255, 255, 255, 0.9);
    }
    
    QDialogButtonBox QPushButton {
        background: rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 8px 16px;
        min-width: 80px;
    }
    
    QDialogButtonBox QPushButton:hover {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.5);
    }
    
    QProgressDialog {
        background: rgba(20, 20, 30, 0.95);
        color: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
    }
    
    QProgressBar {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        text-align: center;
        color: rgba(255, 255, 255, 0.9);
    }
    
    QProgressBar::chunk {
        background: qlineargradient(
            x1: 0, y1: 0, x2: 1, y2: 0,
            stop: 0 #6366F1,
            stop: 1 #8B5CF6
        );
        border-radius: 7px;
    }
    """
    
    app.setStyleSheet(global_style)


def main():
    """Main application entry point"""
    # Create application
    app = QApplication(sys.argv)
    
    # Configure application
    app.setApplicationDisplayName("Spectrogram Analyzer - Modern UI")
    app.setWindowIcon(qta.icon('fa5s.wave-square', color='#8B5CF6'))
    
    # Load fonts and apply styles
    load_fonts()
    apply_global_style(app)
    
    # Create and show main window
    window = ModernMainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    # Import qtawesome for icons
    try:
        import qtawesome as qta
    except ImportError:
        print("Warning: qtawesome not found. Icons will not be displayed.")
        print("Install with: pip install qtawesome")
        
    main()