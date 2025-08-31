#!/usr/bin/env python3
import sys
import os

# Add the parent directory to the Python path so we can import spectrogram_gui modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# Import the modern main window
from spectrogram_gui.gui.modern_main_window import ModernMainWindow

# Construct the absolute path to modern theme
THEME_PATH = os.path.join(
    os.path.dirname(__file__),
    "styles",
    "modern_theme.qss"
)


def main():
    # Enable high DPI support with better font rendering
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create the QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("Spectrogram Analyzer")
    app.setOrganizationName("Modern Audio Tools")
    
    # Set application style
    app.setStyle("Fusion")  # Modern base style
    
    # Set font for better rendering
    font = app.font()
    font.setPointSize(12)  # Increase base font size
    font.setWeight(400)
    app.setFont(font)
    
    # Load modern theme
    try:
        with open(THEME_PATH, 'r') as f:
            theme = f.read()
            app.setStyleSheet(theme)
    except Exception as e:
        print(f"Warning: Could not load theme: {e}")
    
    # Create and show the modern main window
    window = ModernMainWindow()
    window.show()
    
    # Enter the Qt main event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
