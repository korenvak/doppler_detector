# File: run_gui.py

import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    """
    Application entry point. Creates a QApplication,
    instantiates MainWindow, and starts the Qt event loop.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
