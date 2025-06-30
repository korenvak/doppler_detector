import os
import re
from pathlib import Path
from datetime import datetime
import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QPushButton, QLabel, QListWidget, QListWidgetItem,
    QSplitter, QMessageBox, QMenu, QApplication, QFrame,
    QToolButton, QAction, QGraphicsDropShadowEffect, QShortcut,
    QProgressDialog
)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QEvent

import qtawesome as qta

from spectrogram_gui.gui.spectrogram_canvas import SpectrogramCanvas
from spectrogram_gui.gui.event_annotator import EventAnnotator
from spectrogram_gui.gui.detection_manager import DetectionManager
from spectrogram_gui.gui.sound_device_player import SoundDevicePlayer
from spectrogram_gui.gui.filter_dialog import FilterDialog
from spectrogram_gui.gui.filters import CombinedFilterDialog
from spectrogram_gui.gui.fft_stats_dialog import FFTDialog
from spectrogram_gui.gui.gain_dialog import GainDialog
from spectrogram_gui.gui.detector_params_dialog import DetectorParamsDialog
from spectrogram_gui.gui.param_panel import ParamPanel
from spectrogram_gui.gui.spec_settings_dialog import SpectrogramSettingsDialog
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
      - Emits a signal when Delete is pressed
    """

    fileDeleteRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QListWidget.SingleSelection)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.fileDeleteRequested.emit()
        else:
            super().keyPressEvent(event)

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

        self.annotator = EventAnnotator(self.canvas, undo_callback=self.add_undo_action)
        self.detection_manager = DetectionManager()
        self.audio_player = SoundDevicePlayer()
        self.audio_player.prevRequested.connect(self.load_prev_file)
        self.audio_player.nextRequested.connect(self.load_next_file)

        # Default spectrogram parameters
        self.spectrogram_params = {
            "window_size": 4096,
            "overlap": 75,
            "colormap": "magma",
        }

        # Detector instances
        self.detector = DopplerDetector()
        self.detector.spectrogram_params = self.spectrogram_params

        # --- Left pane (file list) ---
        left_frame = QFrame()
        left_frame.setObjectName("card")
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(12,12,12,12)
        left_layout.setSpacing(8)

        heading = QLabel("Loaded Files")
        heading.setObjectName("fileListHeading")

        self.open_files_btn = QToolButton()
        self.open_files_btn.setText("Open Files")
        self.open_files_btn.setIcon(qta.icon('fa5s.folder-open'))
        self.open_files_btn.setPopupMode(QToolButton.InstantPopup)
        open_menu = QMenu(self)
        open_menu.addAction("Open Single File", self.select_multiple_files)
        open_menu.addAction("Open Folder", self.select_folder)
        self.open_files_btn.setMenu(open_menu)

        self.remove_file_btn = QToolButton()
        self.remove_file_btn.setIcon(qta.icon('fa5s.trash'))
        self.remove_file_btn.setToolTip("Remove Selected")
        self.remove_file_btn.clicked.connect(self.remove_selected_file)

        open_layout = QHBoxLayout()
        open_layout.setContentsMargins(0, 0, 0, 0)
        open_layout.addWidget(heading)
        open_layout.addStretch()
        open_layout.addWidget(self.open_files_btn)
        open_layout.addWidget(self.remove_file_btn)
        left_layout.addLayout(open_layout)

        self.file_list = FileListWidget()
        self.file_list.itemClicked.connect(self.load_file)
        self.file_list.fileDeleteRequested.connect(self.remove_selected_file)
        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.on_file_list_context_menu)
        self.file_list.setIconSize(QSize(14, 14))
        left_layout.addWidget(self.file_list, 1)

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
        self.set_csv_btn = QPushButton("Set CSV")
        self.set_csv_btn.setIcon(qta.icon('fa5s.save'))
        self.set_csv_btn.clicked.connect(self.set_csv_file)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setIcon(qta.icon('fa5s.cog'))
        self.settings_btn.clicked.connect(self.open_spectrogram_settings)

        # Auto-Detect
        self.auto_detect_btn = QPushButton("Auto-Detect")
        self.auto_detect_btn.setIcon(qta.icon('fa5s.bolt'))
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
        filter_menu.addAction("High-pass…", lambda: self.open_filter_dialog("highpass"))
        filter_menu.addAction("Low-pass…", lambda: self.open_filter_dialog("lowpass"))
        filter_menu.addAction("Band-pass…", lambda: self.open_filter_dialog("bandpass"))
        filter_menu.addSeparator()
        adaptive_menu = QMenu("Adaptive", self)
        for name in [
            "NLMS",
            "Wiener",
            "Gaussian",
            "Median",
            "Gabor",
            "TV Denoise",
            "Sobel Horizontal",
            "White Top-hat",
            "Frangi",
            "Meijering",
            "Track Follow",
            "Enhance Doppler",
        ]:
            adaptive_menu.addAction(
                f"{name}…",
                lambda n=name: CombinedFilterDialog(self, n).exec_(),
            )
        filter_menu.addMenu(adaptive_menu)
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
        self.status_label = QLabel("")
        right_layout.addWidget(self.status_label)

        # Spectrogram canvas
        self.canvas_container = QFrame()
        self.canvas_container.setObjectName("card")
        canvas_layout = QVBoxLayout(self.canvas_container)
        canvas_layout.setContentsMargins(4,4,4,4)
        canvas_layout.addWidget(self.canvas)
        shadow_c = QGraphicsDropShadowEffect()
        shadow_c.setBlurRadius(16)
        shadow_c.setColor(Qt.black)
        shadow_c.setOffset(0,1)
        self.canvas_container.setGraphicsEffect(shadow_c)
        self.canvas_container.installEventFilter(self)
        right_layout.addWidget(self.canvas_container)

        # Default filter parameters must exist before binding the panel
        self.filter_params = {
            "nlms_enabled": False,
            "lms_enabled": False,
            "ale_enabled": False,
            "rls_enabled": False,
            "wiener_enabled": False,
            "nlms_mu": 10,
            "lms_mu": 10,
            "ale_mu": 10,
            "ale_delay": 0,
            "rls_lambda": 99,
            "wiener_noise": -20,
        }

        self.param_panel = ParamPanel(self)
        self.param_panel.toggle(False)
        self.param_panel.bind_settings()
        right_layout.addWidget(self.param_panel)

        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0,0,0,0)
        controls_layout.setAlignment(Qt.AlignCenter)

        self.dashboard_btn = QToolButton()
        self.dashboard_btn.setObjectName('togglePanel')
        self.dashboard_btn.setCheckable(True)
        self.dashboard_btn.setIcon(qta.icon('fa5s.sliders-h'))
        self.dashboard_btn.setToolTip('Show Detection Dashboard')
        self.dashboard_btn.toggled.connect(self.toggle_param_panel)

        controls_layout.addWidget(self.audio_player)
        controls_layout.addWidget(self.dashboard_btn)
        right_layout.addLayout(controls_layout)
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
        self.undo_stack = []

        # Undo shortcut
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.perform_undo)
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(self.load_prev_file)
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(self.load_next_file)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.open_spectrogram_settings)
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self.run_detection)

        # adjust file list height to match spectrogram
        self.update_file_list_height()

    def add_undo_action(self, action):
        self.undo_stack.append(action)
        self.undo_btn.setEnabled(True)

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
        dlg = DetectorParamsDialog(self, detector=self.detector, mode="peaks")
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
            start_time = datetime.now()
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

                progress = QProgressDialog("Detecting tracks...", "", 0, len(times), self)
                progress.setWindowModality(Qt.WindowModal)
                progress.show()

                peaks  = self.detector.detect_peaks_per_frame()
                tracks = self.detector.track_peaks_over_time(peaks, progress_callback=progress.setValue)
                raw_tracks = self.detector.merge_tracks(tracks)
                progress.close()

            else:
                y, sr = load_audio_with_filters(self.current_file)
                freqs, times, Sxx, _ = compute_spectrogram(
                    y, sr, self.current_file, params=self.spectrogram_params
                )
                self.detector.freqs = freqs
                self.detector.times = times
                self.detector.Sxx_filt = Sxx

                progress = QProgressDialog("Detecting tracks...", "", 0, len(times), self)
                progress.setWindowModality(Qt.WindowModal)
                progress.show()

                peaks = self.detector.detect_peaks_per_frame()
                tracks = self.detector.track_peaks_over_time(peaks, progress_callback=progress.setValue)
                raw_tracks = self.detector.merge_tracks(tracks)
                progress.close()

            # 3) Convert index‐based tracks → time/freq arrays
            processed = []
            for tr in raw_tracks:
                if isinstance(tr, dict):
                    t_idx, f_idx = tr.get("indices", ([], []))
                else:
                    t_idx = [pt[0] for pt in tr]
                    f_idx = [pt[1] for pt in tr]
                t_idx = np.asarray(t_idx, dtype=int)
                f_idx = np.asarray(f_idx, dtype=int)
                times_arr = self.detector.times[t_idx]
                freqs_arr = self.detector.freqs[f_idx]
                processed.append((times_arr, freqs_arr))

            # 4) Clear old and draw new
            self.canvas.clear_auto_tracks()
            self.canvas.plot_auto_tracks(processed)
            self.detection_manager.record(self.canvas.auto_tracks_items.copy())
            self.add_undo_action(("detection", None))
            duration = (datetime.now() - start_time).total_seconds()
            self.param_panel.update_stats(len(processed), "peaks", duration)
            self.status_label.setText(
                f"Auto detection: {len(processed)} tracks found."
            )

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


    def load_file_from_path(self, path, maintain_view=False):
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
        timestamp = parse_timestamp_from_filename(path)
        if timestamp is None:
            try:
                timestamp = datetime.fromtimestamp(os.path.getmtime(path))
            except Exception:
                timestamp = None

        # load audio & spectrogram
        try:
            y, sr = load_audio_with_filters(path)
            freqs, times, Sxx, Sxx_filt = compute_spectrogram(
                y, sr, path, params=self.spectrogram_params
            )
            self.detector.freqs = freqs
            self.detector.times = times
            self.detector.Sxx_filt = Sxx_filt
        except Exception as e:
            QMessageBox.critical(self, "Spectrogram Error", str(e))
            return

        self.canvas.plot_spectrogram(freqs, times, Sxx, timestamp, maintain_view=maintain_view)
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




    def on_change_cmap(self, cmap_name):
        self.spectrogram_params["colormap"] = cmap_name
        self.canvas.set_colormap(cmap_name)


    def on_file_list_context_menu(self, pos):
        menu = QMenu(self.file_list)
        remove_action  = menu.addAction("Remove File")
        menu.addSeparator()
        sort_name_asc  = menu.addAction("Sort by Name (A → Z)")
        sort_name_desc = menu.addAction("Sort by Name (Z → A)")
        sort_date_asc  = menu.addAction("Sort by Date (Old → New)")
        sort_date_desc = menu.addAction("Sort by Date (New → Old)")

        action = menu.exec_(self.file_list.viewport().mapToGlobal(pos))
        if action == remove_action:
            self.remove_selected_file()
        elif action == sort_name_asc:
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


    def remove_selected_file(self):
        item = self.file_list.currentItem()
        if not item:
            return
        row = self.file_list.row(item)
        self.file_list.takeItem(row)
        if item.data(Qt.UserRole) == self.current_file:
            self.current_file = None
            self.info_label.setText("No file loaded")
            self.audio_player.stop()


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
        self.add_undo_action(("waveform", (wave.copy(),
                                Sxx_raw.copy(),
                                times.copy(),
                                freqs.copy(),
                                start_time)))

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
        self.add_undo_action(("waveform", (wave.copy(),
                                Sxx_raw.copy(),
                                times.copy(),
                                freqs.copy(),
                                start_time)))

        dlg = GainDialog(self)
        dlg.exec_()

    def open_spectrogram_settings(self):
        dlg = SpectrogramSettingsDialog(self, params=self.spectrogram_params)
        if dlg.exec_() == dlg.Accepted:
            params = dlg.get_params()
            self.spectrogram_params.update(params)
            if self.current_file:
                self.load_file_from_path(self.current_file, maintain_view=True)

    def toggle_param_panel(self, visible):
        self.param_panel.toggle(visible)
        icon = qta.icon('fa5s.chevron-down') if visible else qta.icon('fa5s.sliders-h')
        self.dashboard_btn.setIcon(icon)

    def update_file_list_height(self):
        if hasattr(self, 'canvas_container'):
            self.file_list.setFixedHeight(self.canvas_container.height())

    def eventFilter(self, obj, event):
        if obj is getattr(self, 'canvas_container', None) and event.type() == QEvent.Resize:
            self.update_file_list_height()
        return super().eventFilter(obj, event)

    def open_detector_params(self):
        dlg = DetectorParamsDialog(self, detector=self.detector, mode="peaks")
        dlg.exec_()

    def load_prev_file(self):
        row = self.file_list.currentRow()
        if row > 0:
            item = self.file_list.item(row-1)
            self.file_list.setCurrentRow(row-1)
            self.load_file(item)

    def load_next_file(self):
        row = self.file_list.currentRow()
        if row < self.file_list.count()-1:
            item = self.file_list.item(row+1)
            self.file_list.setCurrentRow(row+1)
            self.load_file(item)


    def perform_undo(self):
        if not self.undo_stack:
            return
        action, data = self.undo_stack.pop()
        if action == "waveform":
            prev_wave, prev_sxx, prev_times, prev_freqs, prev_start = data
            self.audio_player.replace_waveform(prev_wave)
            self.canvas.plot_spectrogram(prev_freqs, prev_times, prev_sxx, prev_start, maintain_view=True)
        elif action == "annotation":
            self.annotator.undo_last()
        elif action == "detection":
            self.detection_manager.undo_last(self.canvas)
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
