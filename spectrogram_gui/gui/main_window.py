import os
import re
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QPushButton, QLabel, QListWidget, QListWidgetItem,
    QSplitter, QMessageBox, QMenu, QApplication, QFrame,
    QToolButton, QAction, QGraphicsDropShadowEffect, QShortcut,
    QProgressDialog
)
from PySide6.QtGui import QKeySequence
from PySide6.QtCore import Qt, Sig, Signalnal, QSize, QEvent, QSettings

import qtawesome as qta

from doppler_detector.spectrogram_gui.gui.spectrogram_canvas import SpectrogramCanvas
from doppler_detector.spectrogram_gui.gui.event_annotator import EventAnnotator
from doppler_detector.spectrogram_gui.gui.detection_manager import DetectionManager
from doppler_detector.spectrogram_gui.gui.sound_device_player import SoundDevicePlayer
from doppler_detector.spectrogram_gui.gui.filter_dialog import FilterDialog
from doppler_detector.spectrogram_gui.gui.filters import CombinedFilterDialog
from doppler_detector.spectrogram_gui.gui.fft_stats_dialog import FFTDialog
from doppler_detector.spectrogram_gui.gui.gain_dialog import GainDialog
from doppler_detector.spectrogram_gui.gui.detector_params_dialog import DetectorParamsDialog
from doppler_detector.spectrogram_gui.gui.param_panel import ParamPanel
from doppler_detector.spectrogram_gui.gui.spec_settings_dialog import SpectrogramSettingsDialog
from doppler_detector.spectrogram_gui.utils.audio_utils import load_audio_with_filters
from doppler_detector.spectrogram_gui.utils.spectrogram_utils import compute_spectrogram
from doppler_detector.spectrogram_gui.utils.time_parse import parse_times_from_filename
from doppler_detector.spectrogram_gui.utils.auto_detector import DopplerDetector
from doppler_detector.spectrogram_gui.utils.detector_2d import DopplerDetector2D
from doppler_detector.spectrogram_gui.gui.detector_params_dialog_2d import Detector2DParamsDialog
from doppler_detector.spectrogram_gui.utils.worker_thread import ThreadPoolManager, AudioLoadWorker, SpectrogramWorker
from doppler_detector.spectrogram_gui.utils.logger import debug, info, warning, error

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
      - Supports sorting by name, start time, end time, duration
    """

    fileDeleteRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QListWidget.ExtendedSelection)  # Enable multi-select
        
        # Sorting state
        self.sort_key = "name"
        self.sort_ascending = True
        self.settings = QSettings("SpectrogramGUI", "FileList")
        self.load_sort_settings()

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
    
    def load_sort_settings(self):
        """Load sorting preferences from QSettings."""
        self.sort_key = self.settings.value("sort_key", "name")
        self.sort_ascending = self.settings.value("sort_ascending", True, type=bool)
    
    def save_sort_settings(self):
        """Save sorting preferences to QSettings."""
        self.settings.setValue("sort_key", self.sort_key)
        self.settings.setValue("sort_ascending", self.sort_ascending)
    
    def sort_files(self, key="name", ascending=True):
        """Sort the file list by the specified key."""
        from doppler_detector.spectrogram_gui.utils.time_parse import parse_times_from_filename
        
        self.sort_key = key
        self.sort_ascending = ascending
        self.save_sort_settings()
        
        # Extract items with their data
        items = []
        for i in range(self.count()):
            item = self.takeItem(0)
            items.append(item)
        
        # Sort based on key
        def get_sort_value(item):
            path = item.data(Qt.UserRole)
            fname = os.path.basename(path)
            
            if key == "name":
                return fname.lower()
            elif key in ["start_time", "end_time", "duration"]:
                try:
                    _, start_dt, end_dt = parse_times_from_filename(fname)
                    if key == "start_time":
                        return start_dt
                    elif key == "end_time":
                        return end_dt
                    else:  # duration
                        return (end_dt - start_dt).total_seconds()
                except:
                    # If parsing fails, put at end
                    return datetime.max if key != "duration" else float('inf')
            return fname
        
        items.sort(key=get_sort_value, reverse=not ascending)
        
        # Re-add sorted items
        for item in items:
            self.addItem(item)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectrogram GUI with Auto-Detect")
        self.resize(1200, 700)
        
        # Initialize thread pool for background operations
        self.thread_pool = ThreadPoolManager(max_threads=2)

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
        self.detector2d = DopplerDetector2D()
        self.detector2d.spectrogram_params = self.spectrogram_params

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
        # Apply saved sort on startup
        if self.file_list.count() > 0:
            self.file_list.sort_files(self.file_list.sort_key, self.file_list.sort_ascending)
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

        # 2D Auto-Detect
        self.auto_detect_2d_btn = QPushButton("Auto-Detect 2D")
        self.auto_detect_2d_btn.setIcon(qta.icon('fa5s.crosshairs'))
        self.auto_detect_2d_btn.setToolTip("Run 2D peak detection")
        self.auto_detect_2d_btn.clicked.connect(self.run_detection_2d)


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
                lambda n=name: CombinedFilterDialog(self, n).exec(),
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
            self.auto_detect_2d_btn,
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

        # Keyboard shortcuts
        self.setup_shortcuts()

        # adjust file list height to match spectrogram
        self.update_file_list_height()

    def setup_shortcuts(self):
        """Set up keyboard shortcuts for the application."""
        # File operations
        QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(self.select_files)
        QShortcut(QKeySequence("Ctrl+Shift+O"), self).activated.connect(self.select_folder)
        QShortcut(QKeySequence("Delete"), self).activated.connect(self.remove_selected_file)
        
        # Navigation
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(self.load_prev_file)
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(self.load_next_file)
        QShortcut(QKeySequence("Alt+Left"), self).activated.connect(self.load_prev_file)
        QShortcut(QKeySequence("Alt+Right"), self).activated.connect(self.load_next_file)
        
        # Playback
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.toggle_playback)
        
        # Zoom
        QShortcut(QKeySequence("Ctrl+="), self).activated.connect(lambda: self.zoom_canvas(1.2))
        QShortcut(QKeySequence("Ctrl+-"), self).activated.connect(lambda: self.zoom_canvas(0.8))
        QShortcut(QKeySequence("Ctrl+0"), self).activated.connect(self.reset_zoom)
        
        # Tools
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.open_spectrogram_settings)
        QShortcut(QKeySequence("Ctrl+Return"), self).activated.connect(self.run_detection)
        QShortcut(QKeySequence("Ctrl+M"), self).activated.connect(self.toggle_mark_event)
        
        # Edit
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.perform_undo)
        
        # Preferences
        QShortcut(QKeySequence("Ctrl+,"), self).activated.connect(self.open_preferences)
        
        # Help
        QShortcut(QKeySequence("F1"), self).activated.connect(self.show_shortcuts_help)
    
    def toggle_playback(self):
        """Toggle audio playback."""
        if self.audio_player.playing:
            self.audio_player.stop()
        else:
            self.audio_player.play()
    
    def zoom_canvas(self, factor):
        """Zoom the spectrogram canvas."""
        if hasattr(self.canvas, 'vb') and self.canvas.vb:
            xr, yr = self.canvas.vb.viewRange()
            cx = 0.5 * (xr[0] + xr[1])
            new_width = (xr[1] - xr[0]) / factor
            self.canvas.vb.setXRange(cx - new_width/2, cx + new_width/2, padding=0)
    
    def reset_zoom(self):
        """Reset canvas zoom to fit all."""
        if hasattr(self.canvas, 'vb') and self.canvas.vb:
            self.canvas.vb.autoRange()
    
    def open_preferences(self):
        """Open preferences dialog."""
        from doppler_detector.spectrogram_gui.gui.preferences_dialog import PreferencesDialog
        dialog = PreferencesDialog(self)
        dialog.settings_changed.connect(self.apply_preferences)
        dialog.exec()
    
    def apply_preferences(self):
        """Apply preferences from settings."""
        settings = QSettings("SpectrogramGUI", "Preferences")
        
        # Apply performance settings
        use_opengl = settings.value("performance/opengl", False, type=bool)
        antialiasing = settings.value("performance/antialiasing", False, type=bool)
        self.canvas.set_performance_mode(use_opengl, antialiasing)
        
        # Apply thread pool settings
        thread_count = settings.value("performance/thread_count", 2, type=int)
        self.thread_pool.pool.setMaxThreadCount(thread_count)
        
        # Apply theme settings
        colormap = settings.value("theme/colormap", "magma")
        self.spectrogram_params["colormap"] = colormap
        if hasattr(self.canvas, 'set_colormap'):
            self.canvas.set_colormap(colormap)
    
    def show_shortcuts_help(self):
        """Show keyboard shortcuts help dialog."""
        shortcuts_text = """
        <h3>Keyboard Shortcuts</h3>
        <table>
        <tr><td><b>File Operations:</b></td><td></td></tr>
        <tr><td>Ctrl+O</td><td>Open files</td></tr>
        <tr><td>Ctrl+Shift+O</td><td>Open folder</td></tr>
        <tr><td>Delete</td><td>Remove selected file</td></tr>
        <tr><td>&nbsp;</td><td></td></tr>
        <tr><td><b>Navigation:</b></td><td></td></tr>
        <tr><td>← / Alt+←</td><td>Previous file</td></tr>
        <tr><td>→ / Alt+→</td><td>Next file</td></tr>
        <tr><td>&nbsp;</td><td></td></tr>
        <tr><td><b>Playback:</b></td><td></td></tr>
        <tr><td>Space</td><td>Play/Pause</td></tr>
        <tr><td>&nbsp;</td><td></td></tr>
        <tr><td><b>Zoom:</b></td><td></td></tr>
        <tr><td>Ctrl+=</td><td>Zoom in</td></tr>
        <tr><td>Ctrl+-</td><td>Zoom out</td></tr>
        <tr><td>Ctrl+0</td><td>Reset zoom</td></tr>
        <tr><td>&nbsp;</td><td></td></tr>
        <tr><td><b>Tools:</b></td><td></td></tr>
        <tr><td>Ctrl+S</td><td>Spectrogram settings</td></tr>
        <tr><td>Ctrl+Enter</td><td>Run detection</td></tr>
        <tr><td>Ctrl+M</td><td>Toggle mark event</td></tr>
        <tr><td>Ctrl+Z</td><td>Undo</td></tr>
        <tr><td>&nbsp;</td><td></td></tr>
        <tr><td><b>Other:</b></td><td></td></tr>
        <tr><td>Ctrl+,</td><td>Preferences</td></tr>
        <tr><td>F1</td><td>Show this help</td></tr>
        </table>
        """
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts_text)
    
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
        if dlg.exec() != dlg.Accepted:
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

    def run_detection_2d(self):
        """Run the 2D peak detector and plot tracks."""
        if not self.current_file:
            return

        dlg = Detector2DParamsDialog(self, detector=self.detector2d)
        if dlg.exec() != dlg.Accepted:
            return

        use_filtered = False
        if self.undo_stack:
            resp = QMessageBox.question(
                self,
                "Auto-Detect",
                "Filters have been applied. Detect on the filtered signal?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            use_filtered = resp == QMessageBox.Yes

        try:
            start_time = datetime.now()
            if use_filtered:
                y, sr = self.audio_player.get_waveform_copy(return_sr=True)
                freqs, times, Sxx, _ = compute_spectrogram(
                    y, sr, self.current_file, params=self.spectrogram_params
                )
            else:
                y, sr = load_audio_with_filters(self.current_file)
                freqs, times, Sxx, _ = compute_spectrogram(
                    y, sr, self.current_file, params=self.spectrogram_params
                )

            progress = QProgressDialog("Detecting tracks...", "", 0, len(times), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            tracks = self.detector2d.run_detection(Sxx, freqs, times, progress_callback=progress.setValue)
            progress.close()
            processed = []
            for tr in tracks:
                t_idx = np.asarray([pt[0] for pt in tr], dtype=int)
                f_idx = np.asarray([pt[1] for pt in tr], dtype=int)
                processed.append((times[t_idx], freqs[f_idx]))

            self.canvas.clear_auto_tracks()
            self.canvas.plot_auto_tracks(processed)
            self.detection_manager.record(self.canvas.auto_tracks_items.copy())
            self.add_undo_action(("detection", None))
            duration = (datetime.now() - start_time).total_seconds()
            self.param_panel.update_stats(len(processed), "2D", duration)
            self.status_label.setText(
                f"Auto detection 2D: {len(processed)} tracks found."
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

        # parse site/pixel with robust fallback
        site = fname[:1] if fname else "?"
        
        # Try to parse pixel, start, and end times from filename
        pixel = None
        start_timestamp = None
        end_timestamp = None
        
        try:
            # Try the new parser that gets both start and end times
            pixel_id, start_dt, end_dt = parse_times_from_filename(fname)
            pixel = pixel_id
            start_timestamp = start_dt
            end_timestamp = end_dt
        except ValueError:
            # Fall back to old parsing method for pixel
            for tok in re.split(r"[^0-9]+", fname):
                if tok.isdigit():
                    try:
                        pixel = int(tok)
                        break
                    except ValueError:
                        pass
            if pixel is None:
                pixel = 0
            
            # If we can't parse timestamps from filename, we'll use a simple time starting at 00:00:00
            # This is better than using current time which would be misleading
            # We'll use epoch start (1970-01-01) as a neutral reference that makes it clear
            # the timestamps are not real
            start_timestamp = datetime(2000, 1, 1, 0, 0, 0)  # Use a fixed reference date
            # We don't know the actual duration, so we'll set end_timestamp later after loading audio
            end_timestamp = None

        # Show loading progress
        self.progress = QProgressDialog(f"Loading {fname}...", "Cancel", 0, 100, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setMinimumDuration(500)  # Only show after 500ms
        
        # Create and submit audio loading worker
        audio_worker = AudioLoadWorker(path)
        audio_worker.signals.progress.connect(self.progress.setValue)
        audio_worker.signals.result.connect(
            lambda result: self._on_audio_loaded(result, start_timestamp, end_timestamp, maintain_view)
        )
        audio_worker.signals.error.connect(self._on_load_error)
        audio_worker.signals.finished.connect(lambda: self.progress.close())
        
        # Cancel loading if progress dialog is cancelled
        self.progress.canceled.connect(audio_worker.cancel)
        
        # Submit to thread pool
        self.thread_pool.submit(audio_worker)
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
        else:
            super().keyPressEvent(event)




    def on_change_cmap(self, cmap_name):
        self.spectrogram_params["colormap"] = cmap_name
        self.canvas.set_colormap(cmap_name)


    def on_file_list_context_menu(self, pos):
        menu = QMenu(self.file_list)
        
        # File operations
        if self.file_list.currentItem():
            mark_event_action = menu.addAction("Mark Event")
            mark_event_action.setIcon(qta.icon('fa5s.map-marker-alt'))
            remove_action = menu.addAction("Remove from View")
            remove_action.setIcon(qta.icon('fa5s.trash'))
            menu.addSeparator()
        else:
            mark_event_action = None
            remove_action = None
        
        # Sorting submenu
        sort_menu = menu.addMenu("Sort by")
        sort_menu.setIcon(qta.icon('fa5s.sort'))
        
        # Add checkable actions for sort criteria
        sort_actions = []
        for key, label in [("name", "Name"), ("start_time", "Start Time"), 
                          ("end_time", "End Time"), ("duration", "Duration")]:
            action = sort_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self.file_list.sort_key == key)
            action.setData(key)
            sort_actions.append(action)
        
        sort_menu.addSeparator()
        
        # Ascending/Descending toggle
        asc_action = sort_menu.addAction("Ascending")
        asc_action.setCheckable(True)
        asc_action.setChecked(self.file_list.sort_ascending)
        desc_action = sort_menu.addAction("Descending")
        desc_action.setCheckable(True)
        desc_action.setChecked(not self.file_list.sort_ascending)
        
        # Show selected only option
        if len(self.file_list.selectedItems()) > 1:
            menu.addSeparator()
            show_selected_action = menu.addAction("Show Only Selected")
            show_selected_action.setIcon(qta.icon('fa5s.eye'))
        else:
            show_selected_action = None

        action = menu.exec_(self.file_list.viewport().mapToGlobal(pos))
        
        if action == remove_action:
            self.remove_selected_file()
        elif action == mark_event_action:
            self.toggle_mark_event()
        elif action in sort_actions:
            key = action.data()
            self.file_list.sort_files(key, self.file_list.sort_ascending)
        elif action == asc_action:
            self.file_list.sort_files(self.file_list.sort_key, True)
        elif action == desc_action:
            self.file_list.sort_files(self.file_list.sort_key, False)
        elif action == show_selected_action:
            # Keep only selected items
            selected_items = self.file_list.selectedItems()
            all_items = []
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                if item not in selected_items:
                    all_items.append(item)
            for item in all_items:
                self.file_list.takeItem(self.file_list.row(item))



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
        dlg.exec()


    def open_fft_dialog(self):

        wave, sr = self.audio_player.get_waveform_copy(return_sr=True)
        if wave is None or sr is None:
            QMessageBox.warning(self, "No Audio", "No audio loaded.")
            return

        dlg = FFTDialog(self)
        dlg.exec()


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
        dlg.exec()

    def open_spectrogram_settings(self):
        dlg = SpectrogramSettingsDialog(self, params=self.spectrogram_params)
        if dlg.exec() == dlg.Accepted:
            params = dlg.get_params()
            self.spectrogram_params.update(params)
            if self.current_file:
                self.load_file_from_path(self.current_file, maintain_view=True)

    def recompute_spectrogram_from_current_wave(self, maintain_view=True):
        """Recompute spectrogram from the audio player's current waveform.
        Keeps view if requested and updates detector inputs.
        """
        wave, sr = self.audio_player.get_waveform_copy(return_sr=True)
        if wave is None or sr is None:
            return
        freqs, times, Sxx, Sxx_filt = compute_spectrogram(
            wave, sr, self.current_file or "", params=self.spectrogram_params
        )
        self.canvas.plot_spectrogram(freqs, times, Sxx, self.canvas.start_time, maintain_view=maintain_view)
        self.detector.freqs = freqs
        self.detector.times = times
        self.detector.Sxx_filt = Sxx_filt

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
        dlg.exec()

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
    
    def _on_audio_loaded(self, result, start_timestamp, end_timestamp, maintain_view):
        """Callback when audio is loaded successfully."""
        y, sr, path = result
        fname = os.path.basename(path)
        
        info(f"Audio loaded, computing spectrogram for {fname}")
        
        # Now compute spectrogram in background
        spec_worker = SpectrogramWorker(y, sr, path, self.spectrogram_params)
        spec_worker.signals.progress.connect(lambda v: self.progress.setValue(50 + v//2))
        spec_worker.signals.result.connect(
            lambda spec_result: self._on_spectrogram_computed(
                spec_result, y, sr, path, start_timestamp, end_timestamp, maintain_view
            )
        )
        spec_worker.signals.error.connect(self._on_load_error)
        
        self.thread_pool.submit(spec_worker)
    
    def _on_spectrogram_computed(self, result, y, sr, path, start_timestamp, end_timestamp, maintain_view):
        """Callback when spectrogram is computed successfully."""
        freqs, times, Sxx, _ = result
        fname = os.path.basename(path)
        
        # Update detector
        self.detector.freqs = freqs
        self.detector.times = times
        self.detector.Sxx_filt = Sxx
        
        # If we don't have an end_timestamp, calculate it from audio duration
        if end_timestamp is None and start_timestamp is not None and len(times) > 0:
            duration_seconds = times[-1]
            end_timestamp = start_timestamp + timedelta(seconds=duration_seconds)
        
        # Parse metadata from filename
        site = fname[:1] if fname else "?"
        try:
            pixel_id, _, _ = parse_times_from_filename(fname)
            pixel = pixel_id
        except:
            pixel = 0
        
        # Update UI
        self.canvas.plot_spectrogram(freqs, times, Sxx, start_timestamp, end_timestamp, maintain_view=maintain_view)
        self.audio_player.load_audio(y, sr, path)
        self.canvas.update_playback_position(0)
        
        # Update param panel info
        self.param_panel.update_info(site, pixel, start_timestamp, end_timestamp, path)
        
        info(f"Successfully loaded {fname}")
    
    def _on_load_error(self, error_tuple):
        """Callback when loading fails."""
        exctype, value, tb_str = error_tuple
        error(f"Failed to load file: {value}")
        QMessageBox.critical(self, "Loading Error", f"Failed to load file:\n{value}")


def main():
    import sys

    app = QApplication(sys.argv)

    # Load our custom style.qss
    try:
        with open(STYLE_SHEET_PATH, 'r') as f:
            custom = f.read()
    except Exception:
        custom = ""

    # Apply custom styles
    app.setStyleSheet(custom)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
