import os
import re
from pathlib import Path
from datetime import datetime
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QPushButton, QLabel, QListWidget, QListWidgetItem,
    QSplitter, QMessageBox, QMenu, QApplication, QFrame,
    QToolButton, QAction, QGraphicsDropShadowEffect, QShortcut
)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt

import qtawesome as qta

from spectrogram_gui.gui.spectrogram_canvas import SpectrogramCanvas
from spectrogram_gui.gui.event_annotator import EventAnnotator
from spectrogram_gui.gui.sound_device_player import SoundDevicePlayer
from spectrogram_gui.gui.filter_dialog import FilterDialog
from spectrogram_gui.gui.filters import CombinedFilterDialog
from spectrogram_gui.gui.fft_stats_dialog import FFTDialog
from spectrogram_gui.gui.gain_dialog import GainDialog
from spectrogram_gui.gui.params_dialog import ParamsDialog
from spectrogram_gui.gui.detector_params_dialog import DetectorParamsDialog
from spectrogram_gui.utils.audio_utils import load_audio_with_filters
from spectrogram_gui.utils.spectrogram_utils import (
    compute_spectrogram,
    parse_timestamp_from_filename,
)
from spectrogram_gui.utils.auto_detector import DopplerDetector

# Path to the custom QSS file
STYLE_SHEET_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,  # move up from gui/ to spectrogram_gui/
    "styles", "style.qss"
)


class FileListWidget(QListWidget):
    """
    Subclass of QListWidget that:
      - Accepts external drag-drop of .wav/.flac
      - Provides internal drag-drop reordering
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QListWidget.SingleSelection)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                local_path = url.toLocalFile()
                if os.path.isfile(local_path) and local_path.lower().endswith((".wav", ".flac")):
                    if not any(self.item(i).data(Qt.UserRole) == local_path
                               for i in range(self.count())):
                        item = QListWidgetItem(os.path.basename(local_path))
                        item.setIcon(qta.icon('fa5s.music'))
                        item.setData(Qt.UserRole, local_path)
                        self.addItem(item)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectrogram GUI with Auto-Detect")
        self.resize(1200, 700)

        # Spectrogram canvas & annotator
        self.canvas = SpectrogramCanvas(self)
        self.canvas.hover_callback = self.update_hover_info
        self.canvas.click_callback = self.seek_from_click

        self.annotator = EventAnnotator(self.canvas)
        self.audio_player = SoundDevicePlayer()

        # --- Left pane (file list) ---
        left_frame = QFrame()
        left_frame.setObjectName("card")
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(12,12,12,12)
        left_layout.setSpacing(8)

        heading = QLabel("Open Files")
        heading.setObjectName("fileListHeading")
        left_layout.addWidget(heading)

        self.file_list = FileListWidget()
        self.file_list.itemClicked.connect(self.load_file)
        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.on_file_list_context_menu)
        left_layout.addWidget(self.file_list)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(Qt.black)
        shadow.setOffset(0,2)
        left_frame.setGraphicsEffect(shadow)

        # --- Right pane (toolbar + canvas + controls) ---
        right_frame = QFrame()
        right_frame.setObjectName("card")
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(12,12,12,12)
        right_layout.setSpacing(8)

        # Toolbar buttons
        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.setIcon(qta.icon('fa5s.folder'))
        self.open_folder_btn.clicked.connect(self.select_folder)

        self.open_file_btn = QPushButton("Open File")
        self.open_file_btn.setIcon(qta.icon('fa5s.file'))
        self.open_file_btn.clicked.connect(self.select_multiple_files)

        self.change_cmap_btn = QPushButton("Colormap")
        self.change_cmap_btn.setIcon(qta.icon('fa5s.palette'))
        cmap_menu = QMenu(self)
        for cmap in ["gray","viridis","magma","inferno","plasma"]:
            act = cmap_menu.addAction(cmap.capitalize())
            act.triggered.connect(lambda _, n=cmap: self.on_change_cmap(n))
        self.change_cmap_btn.setMenu(cmap_menu)

        self.set_csv_btn = QPushButton("Set CSV")
        self.set_csv_btn.setIcon(qta.icon('fa5s.save'))
        self.set_csv_btn.clicked.connect(self.set_csv_file)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setIcon(qta.icon('fa5s.cog'))
        self.settings_btn.clicked.connect(self.open_settings_dialog)

        # Auto-Detect
        self.auto_detect_btn = QPushButton("Auto-Detect")
        self.auto_detect_btn.setToolTip("Open parameters and run auto-detection")
        self.auto_detect_btn.clicked.connect(self.run_detection)

        # Mark Event
        self.mark_event_btn = QPushButton("Mark Event")
        self.mark_event_btn.setIcon(qta.icon('fa5s.map-marker-alt'))
        self.mark_event_btn.setCheckable(True)
        self.mark_event_btn.clicked.connect(self.toggle_mark_event)

        # Filter popup (high-pass, band-pass, adaptive)
        self.filter_btn = QToolButton()
        self.filter_btn.setText("Filter")
        self.filter_btn.setIcon(qta.icon('fa5s.filter'))
        self.filter_btn.setPopupMode(QToolButton.InstantPopup)
        filter_menu = QMenu(self)
        filter_menu.addAction("High-pass…",   lambda: FilterDialog(self, mode="highpass").exec_())
        filter_menu.addAction("Band-pass…",   lambda: FilterDialog(self, mode="bandpass").exec_())
        filter_menu.addSeparator()
        filter_menu.addAction("Adaptive Filters…", lambda: CombinedFilterDialog(self).exec_())
        self.filter_btn.setMenu(filter_menu)

        # FFT, Gain, Undo
        self.fft_btn = QPushButton("FFT")
        self.fft_btn.setIcon(qta.icon('fa5s.wave-square'))
        self.fft_btn.clicked.connect(self.open_fft_dialog)

        self.gain_btn = QPushButton("Gain ×2")
        self.gain_btn.setIcon(qta.icon('fa5s.volume-up'))
        self.gain_btn.clicked.connect(self.open_gain_dialog)

        self.undo_btn = QPushButton("Undo (Ctrl+Z)")
        self.undo_btn.setIcon(qta.icon('fa5s.undo'))
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self.perform_undo)

        # Assemble toolbar
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)
        top_bar.setContentsMargins(0,0,0,0)
        for w in [
            self.open_folder_btn,
            self.open_file_btn,
            self.change_cmap_btn,
            self.set_csv_btn,
            self.settings_btn,
            self.auto_detect_btn,
        ]:
            top_bar.addWidget(w)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setFixedHeight(24)
        top_bar.addWidget(sep)

        for w in [
            self.mark_event_btn,
            self.filter_btn,
            self.fft_btn,
            self.gain_btn,
            self.undo_btn
        ]:
            top_bar.addWidget(w)

        self.count_label = QLabel("Total annotations: 0")
        self.hover_info_label = QLabel("Hover over spectrogram for details")
        self.hover_info_label.setObjectName("hoverInfo")
        self.hover_info_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        top_bar.addStretch()
        top_bar.addWidget(self.count_label)
        top_bar.addWidget(self.hover_info_label)

        right_layout.addLayout(top_bar)

        # Info label
        self.info_label = QLabel("No file loaded")
        right_layout.addWidget(self.info_label)

        # Spectrogram canvas
        canvas_container = QFrame()
        canvas_container.setObjectName("card")
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(4,4,4,4)
        canvas_layout.addWidget(self.canvas)
        shadow_c = QGraphicsDropShadowEffect()
        shadow_c.setBlurRadius(16)
        shadow_c.setColor(Qt.black)
        shadow_c.setOffset(0,1)
        canvas_container.setGraphicsEffect(shadow_c)
        right_layout.addWidget(canvas_container)

        # Audio controls
        right_layout.addWidget(self.audio_player)
        shadow_r = QGraphicsDropShadowEffect()
        shadow_r.setBlurRadius(20)
        shadow_r.setColor(Qt.black)
        shadow_r.setOffset(0,2)
        right_frame.setGraphicsEffect(shadow_r)

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_frame)
        splitter.addWidget(right_frame)
        splitter.setStretchFactor(1,1)

        # Main container
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(4,4,4,4)
        main_layout.setSpacing(0)
        main_layout.addWidget(splitter)

        self.setCentralWidget(container)

        # State
        self.audio_folder = None
        self.current_file = None
        self.csv_path = None
        self.spectrogram_params = {
            "window_size": 4096,
            "overlap": 75,
            "colormap": "magma"
        }
        self.undo_stack = []

        # Detector
        self.detector = DopplerDetector()
        self.detector.spectrogram_params = self.spectrogram_params

        # Undo shortcut
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.perform_undo)

    def run_detection(self):
        """
        1) Show DetectorParamsDialog.
        2) Optionally rerun spectrogram on filtered audio if filters applied.
        3) Run DopplerDetector (either full pipeline or just peak/track on filtered Sxx).
        4) Overlay tracks.
        """
        if not self.current_file:
            return

        # 1) Show parameters dialog
        dlg = DetectorParamsDialog(self, detector=self.detector)
        if dlg.exec_() != dlg.Accepted:
            return

        # 2) If filters have been applied, ask which spectrogram to use
        use_filtered = False
        if self.undo_stack:
            resp = QMessageBox.question(
                self,
                "Auto-Detect",
                "Filters have been applied. Detect on the filtered signal?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            use_filtered = (resp == QMessageBox.Yes)

        try:
            if use_filtered:
                # grab filtered waveform & sample rate
                y, sr = self.audio_player.get_waveform_copy(return_sr=True)
                # recompute spectrogram on filtered audio
                freqs, times, Sxx, _ = compute_spectrogram(
                    y, sr, self.current_file, params=self.spectrogram_params
                )
                # seed the detector with this Sxx
                self.detector.freqs    = freqs
                self.detector.times    = times
                self.detector.Sxx_filt = Sxx

                if self.detector.detection_method == "threshold":
                    tracks = self.detector.detect_tracks_by_threshold()
                    raw_tracks = self.detector.merge_tracks(tracks)
                else:
                    peaks  = self.detector.detect_peaks_per_frame()
                    tracks = self.detector.track_peaks_over_time(peaks)
                    raw_tracks = self.detector.merge_tracks(tracks)

            else:
                if self.detector.detection_method == "threshold":
                    raw_tracks = self.detector.run_threshold_detection(self.current_file)
                else:
                    raw_tracks = self.detector.run_detection(self.current_file)

            # 3) Convert index‐based tracks → time/freq arrays
            processed = []
            for tr in raw_tracks:
                t_idx = np.array([pt[0] for pt in tr], dtype=int)
                f_idx = np.array([pt[1] for pt in tr], dtype=int)
                times_arr = self.detector.times[t_idx]
                freqs_arr = self.detector.freqs[f_idx]
                processed.append((times_arr, freqs_arr))

            # 4) Clear old and draw new
            self.canvas.clear_auto_tracks()
            self.canvas.plot_auto_tracks(processed)

        except Exception as e:
            QMessageBox.warning(self, "Auto-Detect Error", str(e))

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        self.audio_folder = folder
        self.populate_file_list()


    def select_multiple_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", "", "Audio Files (*.flac *.wav)"
        )
        if not paths:
            return

        for p in paths:
            full = str(Path(p).resolve())
            if not any(self.file_list.item(i).data(Qt.UserRole) == full
                       for i in range(self.file_list.count())):
                item = QListWidgetItem(os.path.basename(full))
                item.setIcon(qta.icon('fa5s.music'))
                item.setData(Qt.UserRole, full)
                self.file_list.addItem(item)


    def populate_file_list(self):
        self.file_list.clear()
        for root, _, files in os.walk(self.audio_folder or ""):
            for fname in sorted(files):
                if fname.lower().endswith((".flac", ".wav")):
                    full = str(Path(root, fname).resolve())
                    item = QListWidgetItem(fname)
                    item.setIcon(qta.icon('fa5s.music'))
                    item.setData(Qt.UserRole, full)
                    self.file_list.addItem(item)


    def load_file(self, item):
        self.load_file_from_path(item.data(Qt.UserRole))


    def load_file_from_path(self, path):
        self.canvas.clear_annotations()
        fname = os.path.basename(path)
        self.current_file = path

        # parse site/pixel
        try:
            site = fname[0]
            pixel = int(next(p for p in fname.split() if p.isdigit()))
        except StopIteration:
            QMessageBox.critical(self, "Parse Error", "Cannot extract site/pixel from filename.")
            return

        # parse timestamp from filename or fall back to file modification time
        timestamp = parse_timestamp_from_filename(fname)
        if timestamp is None:
            try:
                timestamp = datetime.fromtimestamp(os.path.getmtime(path))
            except Exception:
                timestamp = None

        # load audio & spectrogram
        try:
            y, sr = load_audio_with_filters(path)
            freqs, times, Sxx, _ = compute_spectrogram(y, sr, path, params=self.spectrogram_params)
        except Exception as e:
            QMessageBox.critical(self, "Spectrogram Error", str(e))
            return

        self.canvas.plot_spectrogram(freqs, times, Sxx, timestamp)
        self.canvas.set_colormap(self.spectrogram_params["colormap"])
        self.annotator.set_metadata(site=site, pixel=pixel, file_start=timestamp)

        self.audio_player.load(path)
        self.audio_player.set_position_callback(self.canvas.update_playback_position)
        self.canvas.set_start_time(timestamp)

        self.info_label.setText(f"Loaded {fname} — Pixel: {pixel}, Site: {site}")
        self.undo_stack.clear()
        self.undo_btn.setEnabled(False)


    def set_csv_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotation CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return
        self.csv_path = path
        self.annotator.set_csv_path(path)
        self.count_label.setText(f"Total annotations: {self.annotator.count()}")


    def toggle_mark_event(self, checked):
        if checked:
            self.canvas.click_callback = self.annotator.on_click
            self.mark_event_btn.setText("Mark Event (ON)")
        else:
            self.canvas.click_callback = self.seek_from_click
            self.mark_event_btn.setText("Mark Event")


    def seek_from_click(self, rel_sec, event):
        ms = int(rel_sec * 1000)
        self.audio_player.seek(ms)
        self.canvas.update_playback_position(ms)


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self.audio_player.playing:
                self.audio_player.stop()
            else:
                self.audio_player.play()


    def open_settings_dialog(self):
        dlg = ParamsDialog(self, current_params=self.spectrogram_params)
        if dlg.exec_():
            self.spectrogram_params = dlg.get_params()
            if self.current_file:
                self.load_file_from_path(self.current_file)


    def on_change_cmap(self, cmap_name):
        self.spectrogram_params["colormap"] = cmap_name
        self.canvas.set_colormap(cmap_name)


    def on_file_list_context_menu(self, pos):
        menu = QMenu(self.file_list)
        sort_name_asc  = menu.addAction("Sort by Name (A → Z)")
        sort_name_desc = menu.addAction("Sort by Name (Z → A)")
        sort_date_asc  = menu.addAction("Sort by Date (Old → New)")
        sort_date_desc = menu.addAction("Sort by Date (New → Old)")

        action = menu.exec_(self.file_list.viewport().mapToGlobal(pos))
        if action == sort_name_asc:
            self.sort_file_list_by_name(ascending=True)
        elif action == sort_name_desc:
            self.sort_file_list_by_name(ascending=False)
        elif action == sort_date_asc:
            self.sort_file_list_by_date(ascending=True)
        elif action == sort_date_desc:
            self.sort_file_list_by_date(ascending=False)


    def sort_file_list_by_name(self, ascending=True):
        items = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            name, data = item.text(), item.data(Qt.UserRole)
            items.append((name.lower(), name, data))
        items.sort(key=lambda x: x[0], reverse=not ascending)

        self.file_list.clear()
        for _, name, data in items:
            new_item = QListWidgetItem(name)
            new_item.setIcon(qta.icon('fa5s.music'))
            new_item.setData(Qt.UserRole, data)
            self.file_list.addItem(new_item)


    def sort_file_list_by_date(self, ascending=True):
        items = []
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            name, full = item.text(), item.data(Qt.UserRole)
            try:
                ts = os.path.getmtime(full)
            except Exception:
                ts = 0
            items.append((ts, name, full))
        items.sort(key=lambda x: x[0], reverse=not ascending)

        self.file_list.clear()
        for _, name, full in items:
            new_item = QListWidgetItem(name)
            new_item.setIcon(qta.icon('fa5s.music'))
            new_item.setData(Qt.UserRole, full)
            self.file_list.addItem(new_item)


    def update_hover_info(self, text):
        self.hover_info_label.setText(text)


    def open_filter_dialog(self, filter_type="bandpass"):
        # allow filtering entire spectrogram if no range selected

        wave, sr = self.audio_player.get_waveform_copy(return_sr=True)
        if wave is None or sr is None:
            QMessageBox.warning(self, "No Audio", "No audio loaded.")
            return

        freqs = self.canvas.freqs
        times = self.canvas.times
        Sxx_raw = self.canvas.Sxx_raw
        start_time = self.canvas.start_time

        # Push current state to the undo stack
        self.undo_stack.append((wave.copy(),
                                Sxx_raw.copy(),
                                times.copy(),
                                freqs.copy(),
                                start_time))
        self.undo_btn.setEnabled(True)

        dlg = FilterDialog(self, mode=filter_type)
        dlg.exec_()


    def open_fft_dialog(self):

        wave, sr = self.audio_player.get_waveform_copy(return_sr=True)
        if wave is None or sr is None:
            QMessageBox.warning(self, "No Audio", "No audio loaded.")
            return

        dlg = FFTDialog(self)
        dlg.exec_()


    def open_gain_dialog(self):
        # apply gain even if no range selected
        wave, sr = self.audio_player.get_waveform_copy(return_sr=True)
        if wave is None or sr is None:
            QMessageBox.warning(self, "No Audio", "No audio loaded.")
            return

        freqs = self.canvas.freqs
        times = self.canvas.times
        Sxx_raw = self.canvas.Sxx_raw
        start_time = self.canvas.start_time

        # Push current state to the undo stack
        self.undo_stack.append((wave.copy(),
                                Sxx_raw.copy(),
                                times.copy(),
                                freqs.copy(),
                                start_time))
        self.undo_btn.setEnabled(True)

        dlg = GainDialog(self)
        dlg.exec_()


    def perform_undo(self):
        if not self.undo_stack:
            return
        prev_wave, prev_sxx, prev_times, prev_freqs, prev_start = self.undo_stack.pop()
        self.audio_player.replace_waveform(prev_wave)
        self.canvas.plot_spectrogram(prev_freqs, prev_times, prev_sxx, prev_start)
        if not self.undo_stack:
            self.undo_btn.setEnabled(False)


def main():
    import sys
    import qdarkstyle

    app = QApplication(sys.argv)

    # 1) Load qdarkstyle (base dark theme)
    dark = qdarkstyle.load_stylesheet_pyqt5()

    # 2) Load our custom style.qss
    try:
        with open(STYLE_SHEET_PATH, 'r') as f:
            custom = f.read()
    except Exception:
        custom = ""

    # Combine them (qdarkstyle + custom)
    app.setStyleSheet(dark + "\n" + custom)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
