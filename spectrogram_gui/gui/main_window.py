import os
import re
from pathlib import Path
from datetime import datetime
import numpy as np
from typing import Optional, List, Tuple, Any, Union

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

# Configuration Constants
class AppConfig:
    """Application configuration constants"""
    MAX_UNDO_STACK_SIZE = 50
    DEFAULT_WINDOW_SIZE = (1200, 700)
    SUPPORTED_AUDIO_FORMATS = (".wav", ".flac")
    PROGRESS_UPDATE_INTERVAL = 10
    
    # Default spectrogram parameters
    DEFAULT_SPECTROGRAM_PARAMS = {
        "window_size": 4096,
        "overlap": 75,
        "colormap": "magma",
    }
    
    # Default filter parameters
    DEFAULT_FILTER_PARAMS = {
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

# Path to the custom QSS file
STYLE_SHEET_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,  # move up from gui/ to spectrogram_gui/
    "styles", "style.qss"
)


class AudioFileError(Exception):
    """Custom exception for audio file related errors"""
    pass


class SpectrogramError(Exception):
    """Custom exception for spectrogram computation errors"""
    pass


class FileListWidget(QListWidget):
    """
    Enhanced QListWidget with:
      - External drag-drop of .wav/.flac files
      - Internal drag-drop reordering
      - Delete key handling
      - Improved error handling
    """

    fileDeleteRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QListWidget.SingleSelection)

    def keyPressEvent(self, event):
        """Handle delete key events with null checking"""
        if event and event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.fileDeleteRequested.emit()
        else:
            super().keyPressEvent(event)

    def dragEnterEvent(self, event):
        """Handle drag enter events with validation"""
        if event and event.mimeData() and event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        """Handle drag move events with validation"""
        if event and event.mimeData() and event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        """Handle file drop events with robust error handling"""
        if not event or not event.mimeData():
            super().dropEvent(event)
            return
            
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                try:
                    local_path = url.toLocalFile()
                    if (os.path.isfile(local_path) and 
                        local_path.lower().endswith(AppConfig.SUPPORTED_AUDIO_FORMATS)):
                        
                        # Check for duplicates
                        if not any(self._get_item_path(i) == local_path
                                   for i in range(self.count())):
                            item = QListWidgetItem(os.path.basename(local_path))
                            item.setIcon(qta.icon('fa5s.music'))
                            item.setData(Qt.UserRole, local_path)
                            self.addItem(item)
                except Exception as e:
                    print(f"Error processing dropped file: {e}")
            event.acceptProposedAction()
        else:
            super().dropEvent(event)
    
    def _get_item_path(self, index: int) -> Optional[str]:
        """Safely get item path with null checking"""
        item = self.item(index)
        if item:
            return item.data(Qt.UserRole)
        return None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectrogram GUI with Auto-Detect")
        self.resize(*AppConfig.DEFAULT_WINDOW_SIZE)
        
        # Initialize state variables
        self._initialize_state()
        
        # Setup core components
        self._setup_core_components()
        
        # Setup UI
        self._setup_ui()
        
        # Setup shortcuts and final configuration
        self._setup_shortcuts()
        self.update_file_list_height()

    def _initialize_state(self):
        """Initialize application state variables"""
        self.audio_folder: Optional[str] = None
        self.current_file: Optional[str] = None
        self.csv_path: Optional[str] = None
        self.undo_stack: List[Tuple[str, Any]] = []
        
        # Configuration
        self.spectrogram_params = AppConfig.DEFAULT_SPECTROGRAM_PARAMS.copy()
        self.filter_params = AppConfig.DEFAULT_FILTER_PARAMS.copy()
        
    def _setup_core_components(self):
        """Initialize core application components"""
        try:
            # Spectrogram canvas & annotator
            self.canvas = SpectrogramCanvas(self)
            self.canvas.hover_callback = self.update_hover_info
            self.canvas.click_callback = self.seek_from_click

            self.annotator = EventAnnotator(self.canvas, undo_callback=self.add_undo_action)
            self.detection_manager = DetectionManager()
            self.audio_player = SoundDevicePlayer()
            self.audio_player.prevRequested.connect(self.load_prev_file)
            self.audio_player.nextRequested.connect(self.load_next_file)

            # Detector instances
            self.detector = DopplerDetector()
            self.detector.spectrogram_params = self.spectrogram_params
            
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", 
                               f"Failed to initialize core components: {e}")
            raise

    def _setup_ui(self):
        """Setup the complete user interface"""
        # Create main layout components
        left_frame = self._create_file_list_pane()
        right_frame = self._create_main_content_pane()
        
        # Setup splitter and main container
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_frame)
        splitter.addWidget(right_frame)
        splitter.setStretchFactor(1, 1)

        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(0)
        main_layout.addWidget(splitter)

        self.setCentralWidget(container)

    def _create_file_list_pane(self) -> QFrame:
        """Create the left pane with file list and controls"""
        left_frame = QFrame()
        left_frame.setObjectName("card")
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)

        # Header with title and controls
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

        # Header layout
        open_layout = QHBoxLayout()
        open_layout.setContentsMargins(0, 0, 0, 0)
        open_layout.addWidget(heading)
        open_layout.addStretch()
        open_layout.addWidget(self.open_files_btn)
        open_layout.addWidget(self.remove_file_btn)
        left_layout.addLayout(open_layout)

        # File list widget
        self.file_list = FileListWidget()
        self.file_list.itemClicked.connect(self.load_file)
        self.file_list.fileDeleteRequested.connect(self.remove_selected_file)
        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.on_file_list_context_menu)
        self.file_list.setIconSize(QSize(14, 14))
        left_layout.addWidget(self.file_list, 1)

        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(Qt.black)
        shadow.setOffset(0, 2)
        left_frame.setGraphicsEffect(shadow)

        return left_frame

    def _create_main_content_pane(self) -> QFrame:
        """Create the right pane with toolbar, canvas, and controls"""
        right_frame = QFrame()
        right_frame.setObjectName("card")
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(8)

        # Create and add components
        toolbar = self._create_toolbar()
        right_layout.addLayout(toolbar)

        # Info labels
        self.info_label = QLabel("No file loaded")
        right_layout.addWidget(self.info_label)
        self.status_label = QLabel("")
        right_layout.addWidget(self.status_label)

        # Canvas container
        canvas_container = self._create_canvas_container()
        right_layout.addWidget(canvas_container)

        # Parameter panel
        self.param_panel = ParamPanel(self)
        self.param_panel.toggle(False)
        self.param_panel.bind_settings()
        right_layout.addWidget(self.param_panel)

        # Controls layout
        controls_layout = self._create_controls_layout()
        right_layout.addLayout(controls_layout)

        # Add shadow effect
        shadow_r = QGraphicsDropShadowEffect()
        shadow_r.setBlurRadius(20)
        shadow_r.setColor(Qt.black)
        shadow_r.setOffset(0, 2)
        right_frame.setGraphicsEffect(shadow_r)

        return right_frame

    def _create_toolbar(self) -> QHBoxLayout:
        """Create the main toolbar with all buttons"""
        # Create toolbar buttons
        self._create_toolbar_buttons()
        
        # Assemble toolbar layout
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)
        top_bar.setContentsMargins(0, 0, 0, 0)
        
        # Primary buttons
        for btn in [self.set_csv_btn, self.settings_btn, self.auto_detect_btn]:
            top_bar.addWidget(btn)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setFixedHeight(24)
        top_bar.addWidget(sep)

        # Secondary buttons
        for btn in [self.mark_event_btn, self.filter_btn, self.fft_btn, 
                   self.gain_btn, self.undo_btn]:
            top_bar.addWidget(btn)

        # Status labels
        self.count_label = QLabel("Total annotations: 0")
        self.hover_info_label = QLabel("Hover over spectrogram for details")
        self.hover_info_label.setObjectName("hoverInfo")
        self.hover_info_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        top_bar.addStretch()
        top_bar.addWidget(self.count_label)
        top_bar.addWidget(self.hover_info_label)

        return top_bar

    def _create_toolbar_buttons(self):
        """Create all toolbar buttons with improved organization"""
        # CSV and Settings
        self.set_csv_btn = QPushButton("Set CSV")
        self.set_csv_btn.setIcon(qta.icon('fa5s.save'))
        self.set_csv_btn.clicked.connect(self.set_csv_file)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setIcon(qta.icon('fa5s.cog'))
        self.settings_btn.clicked.connect(self.open_spectrogram_settings)

        # Detection
        self.auto_detect_btn = QPushButton("Auto-Detect")
        self.auto_detect_btn.setIcon(qta.icon('fa5s.bolt'))
        self.auto_detect_btn.setToolTip("Open parameters and run auto-detection")
        self.auto_detect_btn.clicked.connect(self.run_detection)

        # Annotation
        self.mark_event_btn = QPushButton("Mark Event")
        self.mark_event_btn.setIcon(qta.icon('fa5s.map-marker-alt'))
        self.mark_event_btn.setCheckable(True)
        self.mark_event_btn.clicked.connect(self.toggle_mark_event)

        # Filter button with menu
        self.filter_btn = self._create_filter_button()

        # Processing buttons
        self.fft_btn = QPushButton("FFT")
        self.fft_btn.setIcon(qta.icon('fa5s.wave-square'))
        self.fft_btn.clicked.connect(self.open_fft_dialog)

        self.gain_btn = QPushButton("Gain ×2")
        self.gain_btn.setIcon(qta.icon('fa5s.volume-up'))
        self.gain_btn.clicked.connect(self.open_gain_dialog)

        # Undo
        self.undo_btn = QPushButton("Undo (Ctrl+Z)")
        self.undo_btn.setIcon(qta.icon('fa5s.undo'))
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self.perform_undo)

    def _create_filter_button(self) -> QToolButton:
        """Create the filter button with its menu"""
        filter_btn = QToolButton()
        filter_btn.setText("Filter")
        filter_btn.setIcon(qta.icon('fa5s.filter'))
        filter_btn.setPopupMode(QToolButton.InstantPopup)
        
        filter_menu = QMenu(self)
        filter_menu.addAction("High-pass…", lambda: self.open_filter_dialog("highpass"))
        filter_menu.addAction("Low-pass…", lambda: self.open_filter_dialog("lowpass"))
        filter_menu.addAction("Band-pass…", lambda: self.open_filter_dialog("bandpass"))
        filter_menu.addSeparator()
        
        adaptive_menu = QMenu("Adaptive", self)
        adaptive_filters = [
            "NLMS", "Wiener", "Gaussian", "Median", "Gabor", "TV Denoise",
            "Sobel Horizontal", "White Top-hat", "Frangi", "Meijering", 
            "Track Follow", "Enhance Doppler"
        ]
        
        for name in adaptive_filters:
            adaptive_menu.addAction(
                f"{name}…",
                lambda n=name: CombinedFilterDialog(self, n).exec_()
            )
        
        filter_menu.addMenu(adaptive_menu)
        filter_btn.setMenu(filter_menu)
        return filter_btn

    def _create_canvas_container(self) -> QFrame:
        """Create the canvas container with shadow effects"""
        self.canvas_container = QFrame()
        self.canvas_container.setObjectName("card")
        canvas_layout = QVBoxLayout(self.canvas_container)
        canvas_layout.setContentsMargins(4, 4, 4, 4)
        canvas_layout.addWidget(self.canvas)
        
        shadow_c = QGraphicsDropShadowEffect()
        shadow_c.setBlurRadius(16)
        shadow_c.setColor(Qt.black)
        shadow_c.setOffset(0, 1)
        self.canvas_container.setGraphicsEffect(shadow_c)
        self.canvas_container.installEventFilter(self)
        
        return self.canvas_container

    def _create_controls_layout(self) -> QHBoxLayout:
        """Create the bottom controls layout"""
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setAlignment(Qt.AlignCenter)

        self.dashboard_btn = QToolButton()
        self.dashboard_btn.setObjectName('togglePanel')
        self.dashboard_btn.setCheckable(True)
        self.dashboard_btn.setIcon(qta.icon('fa5s.sliders-h'))
        self.dashboard_btn.setToolTip('Show Detection Dashboard')
        self.dashboard_btn.toggled.connect(self.toggle_param_panel)

        controls_layout.addWidget(self.audio_player)
        controls_layout.addWidget(self.dashboard_btn)
        
        return controls_layout

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        shortcuts = [
            ("Ctrl+Z", self.perform_undo),
            (Qt.Key_Left, self.load_prev_file),
            (Qt.Key_Right, self.load_next_file),
            ("Ctrl+S", self.open_spectrogram_settings),
            ("Ctrl+Return", self.run_detection)
        ]
        
        for key, slot in shortcuts:
            QShortcut(QKeySequence(key), self).activated.connect(slot)

    def add_undo_action(self, action: Tuple[str, Any]):
        """Add action to undo stack with size limiting"""
        self.undo_stack.append(action)
        
        # Limit undo stack size to prevent memory issues
        if len(self.undo_stack) > AppConfig.MAX_UNDO_STACK_SIZE:
            self.undo_stack.pop(0)  # Remove oldest action
            
        self.undo_btn.setEnabled(True)

    def run_detection(self):
        """
        Run auto-detection with comprehensive error handling and progress tracking.
        
        1) Show DetectorParamsDialog
        2) Optionally rerun spectrogram on filtered audio if filters applied
        3) Run DopplerDetector with progress feedback
        4) Overlay tracks on canvas
        """
        if not self.current_file:
            QMessageBox.information(self, "No File", "Please load an audio file first.")
            return

        # Validate detector state
        if not self.detector:
            QMessageBox.critical(self, "Detector Error", "Detector not initialized.")
            return

        try:
            # 1) Show parameters dialog
            dlg = DetectorParamsDialog(self, detector=self.detector, mode="peaks")
            if dlg.exec_() != dlg.Accepted:
                return

            # 2) Determine if using filtered signal
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

            # 3) Run detection with progress tracking
            start_time = datetime.now()
            self.status_label.setText("Running detection...")
            
            try:
                if use_filtered:
                    raw_tracks = self._run_filtered_detection()
                else:
                    raw_tracks = self._run_standard_detection()
                    
            except Exception as e:
                raise Exception(f"Detection failed: {e}")

            # 4) Process and display results
            processed_tracks = self._process_detection_results(raw_tracks)
            self._display_detection_results(processed_tracks, start_time)

        except Exception as e:
            QMessageBox.warning(self, "Auto-Detect Error", str(e))
            self.status_label.setText("Detection failed")

    def _run_filtered_detection(self) -> List:
        """Run detection on filtered audio data"""
        waveform_data = self.audio_player.get_waveform_copy(return_sr=True)
        if not waveform_data or len(waveform_data) != 2:
            raise Exception("No filtered audio data available")
            
        y, sr = waveform_data
        if y is None or sr is None:
            raise Exception("Invalid audio data")

        self.status_label.setText("Recomputing spectrogram on filtered audio...")
        QApplication.processEvents()
        
        freqs, times, Sxx, _ = compute_spectrogram(
            y, sr, self.current_file, params=self.spectrogram_params
        )
        
        if any(x is None for x in [freqs, times, Sxx]):
            raise Exception("Failed to compute spectrogram on filtered audio")

        # Update detector with filtered data
        self.detector.freqs = freqs
        self.detector.times = times
        self.detector.Sxx_filt = Sxx

        return self._run_detection_pipeline(len(times))

    def _run_standard_detection(self) -> List:
        """Run detection on original audio data"""
        self.status_label.setText("Loading original audio...")
        QApplication.processEvents()
        
        y, sr = load_audio_with_filters(self.current_file)
        if y is None or sr is None:
            raise Exception("Failed to load audio file")
            
        self.status_label.setText("Computing spectrogram...")
        QApplication.processEvents()
        
        freqs, times, Sxx, _ = compute_spectrogram(
            y, sr, self.current_file, params=self.spectrogram_params
        )
        
        if any(x is None for x in [freqs, times, Sxx]):
            raise Exception("Failed to compute spectrogram")

        # Update detector
        self.detector.freqs = freqs
        self.detector.times = times
        self.detector.Sxx_filt = Sxx

        return self._run_detection_pipeline(len(times))

    def _run_detection_pipeline(self, total_frames: int) -> List:
        """Run the core detection pipeline with progress feedback"""
        progress = QProgressDialog("Detecting tracks...", "Cancel", 0, total_frames, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        try:
            self.status_label.setText("Detecting peaks...")
            QApplication.processEvents()
            
            peaks = self.detector.detect_peaks_per_frame()
            
            if progress.wasCanceled():
                raise Exception("Detection cancelled by user")
                
            self.status_label.setText("Tracking peaks...")
            QApplication.processEvents()
            
            tracks = self.detector.track_peaks_over_time(
                peaks, 
                progress_callback=lambda x: progress.setValue(x) if not progress.wasCanceled() else None
            )
            
            if progress.wasCanceled():
                raise Exception("Detection cancelled by user")
                
            self.status_label.setText("Merging tracks...")
            QApplication.processEvents()
            
            raw_tracks = self.detector.merge_tracks(tracks)
            
            return raw_tracks
            
        finally:
            progress.close()

    def _process_detection_results(self, raw_tracks: List) -> List[Tuple]:
        """Convert raw tracks to display format"""
        if not raw_tracks:
            return []
            
        processed = []
        for tr in raw_tracks:
            try:
                if isinstance(tr, dict):
                    t_idx, f_idx = tr.get("indices", ([], []))
                else:
                    t_idx = [pt[0] for pt in tr]
                    f_idx = [pt[1] for pt in tr]
                    
                t_idx = np.asarray(t_idx, dtype=int)
                f_idx = np.asarray(f_idx, dtype=int)
                
                if len(t_idx) > 0 and len(f_idx) > 0:
                    times_arr = self.detector.times[t_idx]
                    freqs_arr = self.detector.freqs[f_idx]
                    processed.append((times_arr, freqs_arr))
                    
            except Exception as e:
                print(f"Warning: Error processing track: {e}")
                
        return processed

    def _display_detection_results(self, processed_tracks: List[Tuple], start_time: datetime):
        """Display detection results on canvas and update UI"""
        try:
            # Clear old tracks and display new ones
            self.canvas.clear_auto_tracks()
            self.canvas.plot_auto_tracks(processed_tracks)
            
            # Record for undo functionality
            if hasattr(self.canvas, 'auto_tracks_items'):
                self.detection_manager.record(self.canvas.auto_tracks_items.copy())
            self.add_undo_action(("detection", None))
            
            # Update statistics
            duration = (datetime.now() - start_time).total_seconds()
            if hasattr(self.param_panel, 'update_stats'):
                self.param_panel.update_stats(len(processed_tracks), "peaks", duration)
                
            # Update status
            self.status_label.setText(f"Detection complete: {len(processed_tracks)} tracks found")
            
        except Exception as e:
            raise Exception(f"Failed to display results: {e}")




    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return
        self.audio_folder = folder
        self.populate_file_list()


    def select_multiple_files(self):
        """Select multiple audio files with improved error handling"""
        try:
            file_filter = f"Audio Files (*{' *'.join(AppConfig.SUPPORTED_AUDIO_FORMATS)})"
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Select Audio Files", "", file_filter
            )
            if not paths:
                return

            added_count = 0
            for p in paths:
                try:
                    full_path = str(Path(p).resolve())
                    if self._add_file_to_list(full_path):
                        added_count += 1
                except Exception as e:
                    print(f"Error processing file {p}: {e}")
                    
            if added_count > 0:
                self.status_label.setText(f"Added {added_count} file(s) to list")
                
        except Exception as e:
            QMessageBox.warning(self, "File Selection Error", 
                              f"Error selecting files: {e}")

    def _add_file_to_list(self, file_path: str) -> bool:
        """Add a file to the list if it's not already present"""
        # Check for duplicates
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item and item.data(Qt.UserRole) == file_path:
                return False  # Already exists
        
        # Add new item
        item = QListWidgetItem(os.path.basename(file_path))
        item.setIcon(qta.icon('fa5s.music'))
        item.setData(Qt.UserRole, file_path)
        self.file_list.addItem(item)
        return True


    def populate_file_list(self):
        """Populate file list from selected folder with error handling"""
        if not self.audio_folder:
            return
            
        try:
            self.file_list.clear()
            file_count = 0
            
            for root, _, files in os.walk(self.audio_folder):
                for fname in sorted(files):
                    if fname.lower().endswith(AppConfig.SUPPORTED_AUDIO_FORMATS):
                        try:
                            full_path = str(Path(root, fname).resolve())
                            item = QListWidgetItem(fname)
                            item.setIcon(qta.icon('fa5s.music'))
                            item.setData(Qt.UserRole, full_path)
                            self.file_list.addItem(item)
                            file_count += 1
                        except Exception as e:
                            print(f"Error processing file {fname}: {e}")
            
            self.status_label.setText(f"Loaded {file_count} audio files")
            
        except Exception as e:
            QMessageBox.warning(self, "Folder Loading Error", 
                              f"Error loading files from folder: {e}")

    def load_file(self, item: Optional[QListWidgetItem]):
        """Load file with null checking"""
        if not item:
            return
            
        file_path = item.data(Qt.UserRole)
        if file_path:
            self.load_file_from_path(file_path)


    def load_file_from_path(self, path: str, maintain_view: bool = False):
        """Load audio file with comprehensive error handling"""
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "File Error", "File does not exist or is not accessible.")
            return

        try:
            self.canvas.clear_annotations()
            fname = os.path.basename(path)
            self.current_file = path

            # Parse site/pixel with better error handling
            site, pixel = self._parse_file_metadata(fname)
            if site is None or pixel is None:
                QMessageBox.warning(self, "Parse Warning", 
                                  "Cannot extract site/pixel from filename. Using defaults.")
                site, pixel = "X", 0

            # Parse timestamp with fallback
            timestamp = self._get_file_timestamp(path)

            # Load audio & compute spectrogram with progress indication
            self.status_label.setText("Loading audio...")
            QApplication.processEvents()
            
            try:
                y, sr = load_audio_with_filters(path)
                if y is None or sr is None:
                    raise AudioFileError("Failed to load audio data")
                    
                self.status_label.setText("Computing spectrogram...")
                QApplication.processEvents()
                
                freqs, times, Sxx, Sxx_filt = compute_spectrogram(
                    y, sr, path, params=self.spectrogram_params
                )
                
                if freqs is None or times is None or Sxx is None:
                    raise SpectrogramError("Failed to compute spectrogram")
                    
                # Update detector with new data
                self.detector.freqs = freqs
                self.detector.times = times
                self.detector.Sxx_filt = Sxx_filt
                
            except Exception as e:
                raise SpectrogramError(f"Audio processing failed: {e}")

            # Update UI components
            self.canvas.plot_spectrogram(freqs, times, Sxx, timestamp, maintain_view=maintain_view)
            self.canvas.set_colormap(self.spectrogram_params["colormap"])
            self.annotator.set_metadata(site=site, pixel=pixel, file_start=timestamp)

            # Setup audio player
            self.audio_player.load(path)
            self.audio_player.set_position_callback(self.canvas.update_playback_position)
            self.canvas.set_start_time(timestamp)

            # Update status
            self.info_label.setText(f"Loaded {fname} — Pixel: {pixel}, Site: {site}")
            self.status_label.setText("File loaded successfully")
            
            # Clear undo stack for new file
            self.undo_stack.clear()
            self.undo_btn.setEnabled(False)

        except (AudioFileError, SpectrogramError) as e:
            QMessageBox.critical(self, "Processing Error", str(e))
            self.status_label.setText("Failed to load file")
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", 
                               f"An unexpected error occurred: {e}")
            self.status_label.setText("Failed to load file")

    def _parse_file_metadata(self, filename: str) -> Tuple[Optional[str], Optional[int]]:
        """Parse site and pixel from filename with error handling"""
        try:
            site = filename[0] if filename else None
            
            # Find first digit sequence for pixel
            pixel_match = re.search(r'\d+', filename)
            pixel = int(pixel_match.group()) if pixel_match else None
            
            return site, pixel
        except (IndexError, ValueError, AttributeError):
            return None, None

    def _get_file_timestamp(self, file_path: str) -> Optional[datetime]:
        """Get file timestamp with multiple fallback methods"""
        # Try parsing from filename first
        timestamp = parse_timestamp_from_filename(file_path)
        if timestamp:
            return timestamp
            
        # Fall back to file modification time
        try:
            return datetime.fromtimestamp(os.path.getmtime(file_path))
        except (OSError, ValueError):
            # Last resort: current time
            return datetime.now()


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


    def open_filter_dialog(self, filter_type: str = "bandpass"):
        """Open filter dialog with improved error handling"""
        try:
            # Check if audio is loaded
            if not self.audio_player:
                QMessageBox.warning(self, "No Audio Player", "Audio player not initialized.")
                return
                
            waveform_data = self.audio_player.get_waveform_copy(return_sr=True)
            if not waveform_data or len(waveform_data) != 2:
                QMessageBox.warning(self, "No Audio", "No audio loaded or invalid waveform data.")
                return
                
            wave, sr = waveform_data
            if wave is None or sr is None:
                QMessageBox.warning(self, "No Audio", "No audio loaded.")
                return

            # Get canvas data with null checking
            if not hasattr(self.canvas, 'freqs') or self.canvas.freqs is None:
                QMessageBox.warning(self, "No Spectrogram", "No spectrogram data available.")
                return
                
            freqs = self.canvas.freqs
            times = self.canvas.times
            Sxx_raw = self.canvas.Sxx_raw
            start_time = self.canvas.start_time

            # Validate data before creating undo action
            if any(x is None for x in [freqs, times, Sxx_raw]):
                QMessageBox.warning(self, "Invalid Data", "Spectrogram data is incomplete.")
                return

            # Push current state to the undo stack
            try:
                self.add_undo_action(("waveform", (
                    wave.copy() if hasattr(wave, 'copy') else wave,
                    Sxx_raw.copy() if hasattr(Sxx_raw, 'copy') else Sxx_raw,
                    times.copy() if hasattr(times, 'copy') else times,
                    freqs.copy() if hasattr(freqs, 'copy') else freqs,
                    start_time
                )))
            except Exception as e:
                print(f"Warning: Could not create undo action: {e}")

            # Open filter dialog
            dlg = FilterDialog(self, mode=filter_type)
            dlg.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "Filter Error", 
                               f"Error opening filter dialog: {e}")


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
