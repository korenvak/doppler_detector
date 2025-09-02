#!/usr/bin/env python3
import sys
import os

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor

# Adjust the import path if necessary—this assumes that `main.py`
# lives in `personal/Koren/spectrogram_gui/` alongside folders `gui/`, `styles/`, etc.
from gui.main_window import MainWindow

# Construct the absolute paths to theme files
APP_STYLE_PATH = os.path.join(
    os.path.dirname(__file__),
    "styles",
    "app.qss"  # New modern theme
)

LEGACY_STYLE_PATH = os.path.join(
    os.path.dirname(__file__),
    "styles",
    "style.qss"  # Keep legacy styles for compatibility
)


def create_dark_palette():
    """Create a modern dark palette for the application."""
    palette = QPalette()
    
    # Window colors
    palette.setColor(QPalette.Window, QColor(27, 31, 45))  # Dark blue-gray
    palette.setColor(QPalette.WindowText, QColor(229, 231, 235))  # Light gray text
    
    # Base colors (for input widgets)
    palette.setColor(QPalette.Base, QColor(17, 24, 39))  # Very dark blue-gray
    palette.setColor(QPalette.AlternateBase, QColor(31, 41, 55))  # Slightly lighter
    
    # Text colors
    palette.setColor(QPalette.Text, QColor(229, 231, 235))
    palette.setColor(QPalette.BrightText, Qt.white)
    
    # Button colors
    palette.setColor(QPalette.Button, QColor(55, 65, 81))
    palette.setColor(QPalette.ButtonText, QColor(229, 231, 235))
    
    # Highlight colors (selection)
    palette.setColor(QPalette.Highlight, QColor(79, 70, 229))  # Indigo
    palette.setColor(QPalette.HighlightedText, Qt.white)
    
    # Disabled colors
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(107, 114, 128))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(107, 114, 128))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(107, 114, 128))
    
    # Link colors
    palette.setColor(QPalette.Link, QColor(6, 182, 212))  # Cyan
    palette.setColor(QPalette.LinkVisited, QColor(147, 51, 234))  # Purple
    
    # Tool tips
    palette.setColor(QPalette.ToolTipBase, QColor(31, 41, 55))
    palette.setColor(QPalette.ToolTipText, QColor(229, 231, 235))
    
    return palette


def main():
    # 1) Create the QApplication
    app = QApplication(sys.argv)
    # ensure keyboard events reach viewbox for zoom shortcuts
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # 2) Set dark palette as base (replaces qdarkstyle)
    app.setStyle("Fusion")  # Use Fusion style as base
    dark_palette = create_dark_palette()
    app.setPalette(dark_palette)

    # 3) Load theme files
    styles = []
    
    # Load modern theme
    try:
        with open(APP_STYLE_PATH, 'r') as f:
            styles.append(f.read())
    except Exception:
        pass
    
    # Load legacy styles for compatibility
    try:
        with open(LEGACY_STYLE_PATH, 'r') as f:
            styles.append(f.read())
    except Exception:
        pass

    # 4) Apply combined styles
    app.setStyleSheet("\n".join(styles))

    # 5) Instantiate and show the main window
    window = MainWindow()
    window.show()

    # 6) Enter the Qt main event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
