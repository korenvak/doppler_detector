#!/usr/bin/env python3
import sys
import os

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# If you’re using qdarkstyle, import it Otherwise you can omit these two lines.
import qdarkstyle

# Adjust the import path if necessary—this assumes that `main.py`
# lives in `personal/Koren/spectrogram_gui/` alongside folders `gui/`, `styles/`, etc.
from gui.main_window import MainWindow

# Construct the absolute path to style.qss
STYLE_SHEET_PATH = os.path.join(
    os.path.dirname(__file__),    # personal/Koren/spectrogram_gui/
    "styles",
    "style.qss"
)

# Light theme enhancements (optional)
LIGHT_STYLE_PATH = os.path.join(
    os.path.dirname(__file__),
    "styles",

    "V2style.qss"
)


def main():
    # 1) Create the QApplication
    app = QApplication(sys.argv)
    # ensure keyboard events reach viewbox for zoom shortcuts
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # 2) (Optional) Load qdarkstyle’s base dark theme
    # If you don’t want qdarkstyle, comment out these two lines and
    # change app.setStyleSheet(...) below to app.setStyleSheet(custom)
    dark = qdarkstyle.load_stylesheet_pyqt5()

    # 3) Load our custom QSS (style.qss)
    try:
        with open(STYLE_SHEET_PATH, 'r') as f:
            custom = f.read()
    except Exception:
        custom = ""
    
    # 3.5) Load light theme enhancements (optional, non-breaking)
    try:
        with open(LIGHT_STYLE_PATH, 'r') as f:
            light_enhancements = f.read()
    except Exception:
        light_enhancements = ""

    # 4) Combine qdarkstyle + our custom QSS + light enhancements
    # If you prefer to use ONLY your own style.qss, replace the next line with:
    #     app.setStyleSheet(custom)
    app.setStyleSheet(dark + "\n" + custom + "\n" + light_enhancements)

    # 5) Instantiate and show the main window
    window = MainWindow()
    window.show()

    # 6) Enter the Qt main event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
