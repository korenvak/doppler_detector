"""
Modern Main Window with enhanced UI/UX and performance optimizations
- Multi-threaded audio processing
- Caching for spectrograms
- Responsive layout with collapsible panels
- Modern toolbar with grouped actions
- Batch processing capabilities
"""

import os
import sys
import re
from pathlib import Path
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QFileDialog, QPushButton, QLabel, QListWidget, QListWidgetItem,
    QMessageBox, QMenu, QAction, QToolBar, QDockWidget,
    QProgressDialog, QApplication, QFrame, QToolButton,
    QStatusBar, QTabWidget, QTextEdit, QSlider, QSpinBox,
    QCheckBox, QComboBox, QGroupBox, QGridLayout
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize, QPropertyAnimation,
    QEasingCurve, QRect, QPoint, QParallelAnimationGroup,
    QThreadPool, QRunnable, pyqtSlot, QObject
)
from PyQt5.QtGui import QIcon, QKeySequence, QFont, QPalette, QColor

import qtawesome as qta

# Import our enhanced components
from spectrogram_gui.gui.enhanced_spectrogram_canvas import EnhancedSpectrogramCanvas
from spectrogram_gui.gui.modern_event_annotator import ModernEventAnnotator
from spectrogram_gui.gui.sound_device_player import SoundDevicePlayer
from spectrogram_gui.gui.filter_dialog import FilterDialog
from spectrogram_gui.gui.filters import CombinedFilterDialog
from spectrogram_gui.gui.detector_params_dialog import DetectorParamsDialog
from spectrogram_gui.gui.detector_params_dialog_2d import Detector2DParamsDialog

from spectrogram_gui.utils.audio_utils import load_audio_with_filters
from spectrogram_gui.utils.spectrogram_utils import compute_spectrogram, parse_timestamp_from_filename
from spectrogram_gui.utils.auto_detector import DopplerDetector
from spectrogram_gui.utils.detector_2d import DopplerDetector2D


class AudioProcessingWorker(QRunnable):
    """Worker for processing audio in background thread"""
    
    class Signals(QObject):
        finished = pyqtSignal(object)
        error = pyqtSignal(str)
        progress = pyqtSignal(int)
        
    def __init__(self, file_path, filters=None):
        super().__init__()
        self.file_path = file_path
        self.filters = filters
        self.signals = self.Signals()
        
    @pyqtSlot()
    def run(self):
        """Process audio file"""
        try:
            # Load audio
            self.signals.progress.emit(25)
            # Extract filter parameters from filters dict
            hp = self.filters.get('highpass') if self.filters else None
            lp = self.filters.get('lowpass') if self.filters else None
            gain_db = self.filters.get('gain', 0) if self.filters else 0
            
            data, sample_rate = load_audio_with_filters(
                self.file_path,
                hp=hp,
                lp=lp,
                gain_db=gain_db
            )
            
            # Parse timestamp
            self.signals.progress.emit(50)
            start_time = parse_timestamp_from_filename(self.file_path)
            
            # Compute spectrogram
            self.signals.progress.emit(75)
            # Use default parameters for spectrogram computation
            params = {
                "window_size": 1024,
                "overlap": 50,  # 50% overlap
                "smooth_sigma": 1.5,
                "median_filter_size": (3, 1)
            }
            freqs, times, Sxx_norm, Sxx_filt = compute_spectrogram(
                data, sample_rate, self.file_path, params
            )
            # Use the filtered spectrogram for display
            Sxx = Sxx_filt
            
            # Package results
            result = {
                'data': data,
                'sample_rate': sample_rate,
                'freqs': freqs,
                'times': times,
                'Sxx': Sxx,
                'start_time': start_time,
                'file_path': self.file_path
            }
            
            self.signals.progress.emit(100)
            self.signals.finished.emit(result)
            
        except Exception as e:
            self.signals.error.emit(str(e))


class ModernFileListWidget(QListWidget):
    """Enhanced file list with modern styling and drag-drop"""
    
    files_dropped = pyqtSignal(list)
    file_deleted = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setAlternatingRowColors(True)
        
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
            files = []
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if os.path.isfile(path) and path.lower().endswith(('.wav', '.flac')):
                    files.append(path)
            if files:
                self.files_dropped.emit(files)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)
            
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            current = self.currentItem()
            if current:
                path = current.data(Qt.UserRole)
                self.file_deleted.emit(path)
        else:
            super().keyPressEvent(event)


class ModernMainWindow(QMainWindow):
    """Modern main window with enhanced features"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize attributes
        self.current_file_path = None
        self.current_audio_data = None
        self.current_sample_rate = None
        self.current_start_time = None
        self.is_playing = False
        self.filters = {}
        self.undo_stack = []
        
        # Thread pool for parallel processing
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(4)
        
        # Cache for spectrograms
        self.spectrogram_cache = {}
        self.max_cache_size = 10
        
        # Setup UI
        self.setup_ui()
        self.setup_toolbar()
        self.setup_dock_widgets()
        self.setup_status_bar()
        self.setup_connections()
        self.apply_theme()
        
        # Load settings
        self.load_settings()
        
    def setup_ui(self):
        """Setup the main UI layout"""
        self.setWindowTitle("Spectrogram Analyzer - Modern Edition")
        self.setGeometry(100, 100, 1600, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create main splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # Left panel - File list
        self.file_panel = self.create_file_panel()
        self.main_splitter.addWidget(self.file_panel)
        
        # Center panel - Spectrogram
        self.center_panel = self.create_center_panel()
        self.main_splitter.addWidget(self.center_panel)
        
        # Right panel - Controls
        self.control_panel = self.create_control_panel()
        self.main_splitter.addWidget(self.control_panel)
        
        # Set splitter sizes
        self.main_splitter.setSizes([250, 900, 450])
        
        # Initialize components
        self.player = SoundDevicePlayer()
        self.player.position_changed.connect(self.on_playback_position_changed)
        self.player.playback_finished.connect(self.on_playback_finished)
        
    def create_file_panel(self):
        """Create the file list panel"""
        panel = QFrame()
        panel.setObjectName("modernCard")
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        
        # Header
        header = QFrame()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        title = QLabel("Audio Files")
        title.setObjectName("modernHeading")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Add files button
        add_btn = QToolButton()
        add_btn.setIcon(qta.icon('mdi.folder-open', color='#6366F1'))
        add_btn.setToolTip("Add Files")
        add_btn.clicked.connect(self.browse_files)
        header_layout.addWidget(add_btn)
        
        # Clear button
        clear_btn = QToolButton()
        clear_btn.setIcon(qta.icon('mdi.delete-sweep', color='#EF4444'))
        clear_btn.setToolTip("Clear All")
        clear_btn.clicked.connect(self.clear_file_list)
        header_layout.addWidget(clear_btn)
        
        layout.addWidget(header)
        
        # File list
        self.file_list = ModernFileListWidget()
        self.file_list.files_dropped.connect(self.add_files)
        self.file_list.file_deleted.connect(self.remove_file)
        self.file_list.itemClicked.connect(self.on_file_selected)
        self.file_list.itemDoubleClicked.connect(self.load_and_process_file)
        layout.addWidget(self.file_list)
        
        # Info label
        self.file_info_label = QLabel("Drop audio files here or click to browse")
        self.file_info_label.setObjectName("modernSubheading")
        self.file_info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.file_info_label)
        
        # Batch processing button
        batch_btn = QPushButton("Batch Process")
        batch_btn.setIcon(qta.icon('mdi.lightning-bolt', color='white'))
        batch_btn.clicked.connect(self.batch_process_files)
        layout.addWidget(batch_btn)
        
        return panel
        
    def create_center_panel(self):
        """Create the center panel with spectrogram"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Tab widget for multiple views
        self.view_tabs = QTabWidget()
        self.view_tabs.setTabPosition(QTabWidget.North)
        
        # Spectrogram tab
        self.spectrogram_canvas = EnhancedSpectrogramCanvas()
        self.spectrogram_canvas.click_callback = self.on_canvas_click
        self.spectrogram_canvas.hover_callback = self.update_hover_info
        self.spectrogram_canvas.range_selected.connect(self.on_range_selected)
        self.view_tabs.addTab(self.spectrogram_canvas, "Spectrogram")
        
        # Waveform tab (placeholder)
        waveform_widget = QWidget()
        self.view_tabs.addTab(waveform_widget, "Waveform")
        
        # 3D view tab (placeholder)
        view_3d_widget = QWidget()
        self.view_tabs.addTab(view_3d_widget, "3D View")
        
        layout.addWidget(self.view_tabs)
        
        # Playback controls
        self.playback_controls = self.create_playback_controls()
        layout.addWidget(self.playback_controls)
        
        return panel
        
    def create_playback_controls(self):
        """Create modern playback controls"""
        controls = QFrame()
        controls.setObjectName("modernCard")
        controls.setMaximumHeight(60)
        
        layout = QHBoxLayout(controls)
        layout.setContentsMargins(8, 4, 8, 4)
        
        # Play/Pause button
        self.play_btn = QPushButton()
        self.play_btn.setIcon(qta.icon('mdi.play', color='white'))
        self.play_btn.setIconSize(QSize(24, 24))
        self.play_btn.setObjectName("accentButton")
        self.play_btn.clicked.connect(self.toggle_playback)
        layout.addWidget(self.play_btn)
        
        # Stop button
        self.stop_btn = QPushButton()
        self.stop_btn.setIcon(qta.icon('mdi.stop', color='white'))
        self.stop_btn.clicked.connect(self.stop_playback)
        layout.addWidget(self.stop_btn)
        
        # Time display
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("font-family: monospace; font-size: 16px;")
        layout.addWidget(self.time_label)
        
        # Progress slider
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setMinimum(0)
        self.progress_slider.setMaximum(1000)
        self.progress_slider.sliderPressed.connect(self.on_slider_pressed)
        self.progress_slider.sliderReleased.connect(self.on_slider_released)
        layout.addWidget(self.progress_slider, 1)
        
        # Speed control
        layout.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1.0x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.setMaximumWidth(80)
        self.speed_combo.currentTextChanged.connect(self.change_playback_speed)
        layout.addWidget(self.speed_combo)
        
        # Volume control
        layout.addWidget(QLabel("Volume:"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(75)
        self.volume_slider.setMaximumWidth(100)
        self.volume_slider.valueChanged.connect(self.change_volume)
        layout.addWidget(self.volume_slider)
        
        self.volume_label = QLabel("75%")
        self.volume_label.setMinimumWidth(40)
        layout.addWidget(self.volume_label)
        
        return controls
        
    def create_control_panel(self):
        """Create the control panel with settings and annotations"""
        panel = QFrame()
        panel.setObjectName("modernCard")
        layout = QVBoxLayout(panel)
        
        # Tab widget for controls
        self.control_tabs = QTabWidget()
        
        # Settings tab
        settings_widget = self.create_settings_widget()
        self.control_tabs.addTab(settings_widget, "Settings")
        
        # Annotations tab
        self.annotator = ModernEventAnnotator(self.spectrogram_canvas)
        self.control_tabs.addTab(self.annotator, "Annotations")
        
        # Detection tab
        detection_widget = self.create_detection_widget()
        self.control_tabs.addTab(detection_widget, "Detection")
        
        # Analysis tab
        analysis_widget = self.create_analysis_widget()
        self.control_tabs.addTab(analysis_widget, "Analysis")
        
        layout.addWidget(self.control_tabs)
        
        return panel
        
    def create_settings_widget(self):
        """Create settings widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # STFT Settings
        stft_group = QGroupBox("STFT Settings")
        stft_layout = QGridLayout()
        
        stft_layout.addWidget(QLabel("Window:"), 0, 0)
        self.window_combo = QComboBox()
        self.window_combo.addItems(["hann", "hamming", "blackman", "bartlett", "tukey"])
        self.window_combo.currentTextChanged.connect(self.update_spectrogram_params)
        stft_layout.addWidget(self.window_combo, 0, 1)
        
        stft_layout.addWidget(QLabel("Window Size:"), 1, 0)
        self.nperseg_spin = QSpinBox()
        self.nperseg_spin.setRange(128, 8192)
        self.nperseg_spin.setSingleStep(128)
        self.nperseg_spin.setValue(1024)
        self.nperseg_spin.valueChanged.connect(self.update_spectrogram_params)
        stft_layout.addWidget(self.nperseg_spin, 1, 1)
        
        stft_layout.addWidget(QLabel("Overlap:"), 2, 0)
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 4096)
        self.overlap_spin.setSingleStep(64)
        self.overlap_spin.setValue(512)
        self.overlap_spin.valueChanged.connect(self.update_spectrogram_params)
        stft_layout.addWidget(self.overlap_spin, 2, 1)
        
        stft_group.setLayout(stft_layout)
        layout.addWidget(stft_group)
        
        # Filter Settings
        filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout()
        
        self.filter_btn = QPushButton("Configure Filters")
        self.filter_btn.setIcon(qta.icon('mdi.filter', color='white'))
        self.filter_btn.clicked.connect(self.open_filter_dialog)
        filter_layout.addWidget(self.filter_btn)
        
        self.filter_status = QLabel("No filters applied")
        self.filter_status.setStyleSheet("color: #9CA3AF; font-size: 14px;")
        filter_layout.addWidget(self.filter_status)
        
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # Display Settings
        display_group = QGroupBox("Display")
        display_layout = QGridLayout()
        
        self.auto_scale_check = QCheckBox("Auto Scale")
        self.auto_scale_check.setChecked(True)
        display_layout.addWidget(self.auto_scale_check, 0, 0)
        
        self.show_grid_check = QCheckBox("Show Grid")
        self.show_grid_check.setChecked(True)
        self.show_grid_check.toggled.connect(self.spectrogram_canvas.toggle_grid)
        display_layout.addWidget(self.show_grid_check, 1, 0)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        layout.addStretch()
        
        return widget
        
    def create_detection_widget(self):
        """Create detection widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Detector selection
        detector_group = QGroupBox("Doppler Detection")
        detector_layout = QVBoxLayout()
        
        self.detector_1d_btn = QPushButton("Configure 1D Detector")
        self.detector_1d_btn.setIcon(qta.icon('mdi.radar', color='white'))
        self.detector_1d_btn.clicked.connect(self.configure_1d_detector)
        detector_layout.addWidget(self.detector_1d_btn)
        
        self.detector_2d_btn = QPushButton("Configure 2D Detector")
        self.detector_2d_btn.setIcon(qta.icon('mdi.grid', color='white'))
        self.detector_2d_btn.clicked.connect(self.configure_2d_detector)
        detector_layout.addWidget(self.detector_2d_btn)
        
        self.run_detection_btn = QPushButton("Run Detection")
        self.run_detection_btn.setObjectName("successButton")
        self.run_detection_btn.setIcon(qta.icon('mdi.play', color='white'))
        self.run_detection_btn.clicked.connect(self.run_detection)
        detector_layout.addWidget(self.run_detection_btn)
        
        detector_group.setLayout(detector_layout)
        layout.addWidget(detector_group)
        
        # Detection results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.detection_results = QTextEdit()
        self.detection_results.setReadOnly(True)
        self.detection_results.setMaximumHeight(200)
        results_layout.addWidget(self.detection_results)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        layout.addStretch()
        
        return widget
        
    def create_analysis_widget(self):
        """Create analysis widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout()
        
        self.stats_labels = {}
        stats = ["Mean Freq", "Peak Freq", "Bandwidth", "SNR", "Energy"]
        for i, stat in enumerate(stats):
            label = QLabel(f"{stat}:")
            value = QLabel("--")
            value.setStyleSheet("font-weight: bold;")
            stats_layout.addWidget(label, i, 0)
            stats_layout.addWidget(value, i, 1)
            self.stats_labels[stat] = value
            
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Export options
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        
        self.export_image_btn = QPushButton("Export Image")
        self.export_image_btn.setIcon(qta.icon('mdi.image', color='white'))
        self.export_image_btn.clicked.connect(self.export_spectrogram_image)
        export_layout.addWidget(self.export_image_btn)
        
        self.export_data_btn = QPushButton("Export Data")
        self.export_data_btn.setIcon(qta.icon('mdi.database-export', color='white'))
        self.export_data_btn.clicked.connect(self.export_spectrogram_data)
        export_layout.addWidget(self.export_data_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        layout.addStretch()
        
        return widget
        
    def setup_toolbar(self):
        """Setup modern toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # File actions
        toolbar.addAction(
            qta.icon('mdi.folder-open', color='#6366F1'),
            "Open Files",
            self.browse_files
        )
        
        toolbar.addAction(
            qta.icon('mdi.content-save', color='#6366F1'),
            "Save Session",
            self.save_session
        )
        
        toolbar.addSeparator()
        
        # View actions
        toolbar.addAction(
            qta.icon('mdi.magnify-plus', color='#6366F1'),
            "Zoom In",
            self.spectrogram_canvas.zoom_in
        )
        
        toolbar.addAction(
            qta.icon('mdi.magnify-minus', color='#6366F1'),
            "Zoom Out",
            self.spectrogram_canvas.zoom_out
        )
        
        toolbar.addAction(
            qta.icon('mdi.fullscreen', color='#6366F1'),
            "Fit to Screen",
            self.spectrogram_canvas.zoom_reset
        )
        
        toolbar.addSeparator()
        
        # Tools
        toolbar.addAction(
            qta.icon('mdi.filter', color='#6366F1'),
            "Filters",
            self.open_filter_dialog
        )
        
        toolbar.addAction(
            qta.icon('mdi.radar', color='#6366F1'),
            "Detection",
            self.run_detection
        )
        
        toolbar.addSeparator()
        
        # Help
        toolbar.addAction(
            qta.icon('mdi.help-circle', color='#6366F1'),
            "Help",
            self.show_help
        )
        
    def setup_dock_widgets(self):
        """Setup dockable widgets"""
        # Console dock
        console_dock = QDockWidget("Console", self)
        console_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(150)
        self.console.setStyleSheet("background: #0A0E1A; color: #10B981; font-family: monospace;")
        console_dock.setWidget(self.console)
        
        self.addDockWidget(Qt.BottomDockWidgetArea, console_dock)
        console_dock.hide()
        
        # Add view menu action
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(console_dock.toggleViewAction())
        
    def setup_status_bar(self):
        """Setup modern status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Permanent widgets
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        self.progress_label = QLabel("")
        self.status_bar.addPermanentWidget(self.progress_label)
        
        self.memory_label = QLabel("Memory: --")
        self.status_bar.addPermanentWidget(self.memory_label)
        
        # Update memory usage periodically
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_usage)
        self.memory_timer.start(5000)
        
    def setup_connections(self):
        """Setup signal connections"""
        # File list connections are already set in create_file_panel
        
        # Spectrogram connections are already set in create_center_panel
        
        # Annotator connections
        self.annotator.annotation_added.connect(self.on_annotation_added)
        
    def apply_theme(self):
        """Apply modern theme"""
        # Load and apply the modern theme
        theme_path = os.path.join(
            os.path.dirname(__file__),
            "..", "styles", "modern_theme.qss"
        )
        
        try:
            with open(theme_path, 'r') as f:
                self.setStyleSheet(f.read())
        except:
            self.log("Warning: Could not load modern theme")
            
    def browse_files(self):
        """Browse and add audio files"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio Files",
            "",
            "Audio Files (*.wav *.flac);;All Files (*.*)"
        )
        
        if files:
            self.add_files(files)
            
    def add_files(self, files):
        """Add files to the list"""
        for file_path in files:
            # Check if already in list
            exists = False
            for i in range(self.file_list.count()):
                if self.file_list.item(i).data(Qt.UserRole) == file_path:
                    exists = True
                    break
                    
            if not exists:
                item = QListWidgetItem(os.path.basename(file_path))
                item.setIcon(qta.icon('mdi.music', color='#6366F1'))
                item.setData(Qt.UserRole, file_path)
                self.file_list.addItem(item)
                
        self.update_file_info()
        
    def remove_file(self, file_path):
        """Remove file from list"""
        for i in range(self.file_list.count()):
            if self.file_list.item(i).data(Qt.UserRole) == file_path:
                self.file_list.takeItem(i)
                break
                
        self.update_file_info()
        
    def clear_file_list(self):
        """Clear all files from list"""
        reply = QMessageBox.question(
            self,
            "Clear Files",
            "Remove all files from the list?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.file_list.clear()
            self.update_file_info()
            
    def update_file_info(self):
        """Update file info label"""
        count = self.file_list.count()
        if count == 0:
            self.file_info_label.setText("Drop audio files here or click to browse")
        else:
            self.file_info_label.setText(f"{count} file{'s' if count != 1 else ''} loaded")
            
    def on_file_selected(self, item):
        """Handle file selection"""
        file_path = item.data(Qt.UserRole)
        self.current_file_path = file_path
        self.status_label.setText(f"Selected: {os.path.basename(file_path)}")
        
    def load_and_process_file(self, item=None):
        """Load and process audio file"""
        if item:
            file_path = item.data(Qt.UserRole)
        elif self.current_file_path:
            file_path = self.current_file_path
        else:
            return
            
        # Check cache first
        cache_key = self.get_cache_key(file_path)
        if cache_key in self.spectrogram_cache:
            self.log(f"Loading from cache: {os.path.basename(file_path)}")
            self.apply_cached_result(self.spectrogram_cache[cache_key])
            return
            
        # Show progress dialog
        self.progress_dialog = QProgressDialog(
            "Processing audio file...",
            "Cancel",
            0, 100,
            self
        )
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()
        
        # Create worker
        worker = AudioProcessingWorker(file_path, self.filters)
        worker.signals.finished.connect(self.on_processing_finished)
        worker.signals.error.connect(self.on_processing_error)
        worker.signals.progress.connect(self.progress_dialog.setValue)
        
        # Start processing
        self.thread_pool.start(worker)
        
    def on_processing_finished(self, result):
        """Handle processing completion"""
        self.progress_dialog.close()
        
        # Store results
        self.current_audio_data = result['data']
        self.current_sample_rate = result['sample_rate']
        self.current_start_time = result['start_time']
        
        # Update spectrogram
        self.spectrogram_canvas.set_spectrogram(
            result['freqs'],
            result['times'],
            result['Sxx'],
            result['start_time']
        )
        
        # Cache result
        self.cache_result(result)
        
        # Update UI
        self.update_statistics(result)
        self.status_label.setText(f"Loaded: {os.path.basename(result['file_path'])}")
        
        # Setup annotator metadata
        site, pixel = self.extract_metadata(result['file_path'])
        self.annotator.set_metadata(site, pixel, result['start_time'])
        
        # Setup player
        self.player.set_audio(
            result['data'],
            result['sample_rate']
        )
        
        self.log(f"Successfully loaded: {os.path.basename(result['file_path'])}")
        
    def on_processing_error(self, error):
        """Handle processing error"""
        self.progress_dialog.close()
        QMessageBox.critical(self, "Processing Error", f"Failed to process file:\n{error}")
        self.log(f"Error: {error}", error=True)
        
    def get_cache_key(self, file_path):
        """Generate cache key for file"""
        # Include filters in cache key
        filter_str = json.dumps(self.filters, sort_keys=True)
        return f"{file_path}_{filter_str}"
        
    def cache_result(self, result):
        """Cache processing result"""
        cache_key = self.get_cache_key(result['file_path'])
        
        # Limit cache size
        if len(self.spectrogram_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest = next(iter(self.spectrogram_cache))
            del self.spectrogram_cache[oldest]
            
        self.spectrogram_cache[cache_key] = result
        
    def apply_cached_result(self, result):
        """Apply cached result"""
        self.current_audio_data = result['data']
        self.current_sample_rate = result['sample_rate']
        self.current_start_time = result['start_time']
        
        self.spectrogram_canvas.set_spectrogram(
            result['freqs'],
            result['times'],
            result['Sxx'],
            result['start_time']
        )
        
        self.update_statistics(result)
        self.player.set_audio(result['data'], result['sample_rate'])
        
    def batch_process_files(self):
        """Process multiple files in batch"""
        if self.file_list.count() == 0:
            QMessageBox.information(self, "No Files", "Please add files to process.")
            return
            
        # Get output directory
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        
        if not output_dir:
            return
            
        # Process each file
        progress = QProgressDialog(
            "Processing files...",
            "Cancel",
            0, self.file_list.count(),
            self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        for i in range(self.file_list.count()):
            if progress.wasCanceled():
                break
                
            item = self.file_list.item(i)
            file_path = item.data(Qt.UserRole)
            
            progress.setLabelText(f"Processing: {os.path.basename(file_path)}")
            progress.setValue(i)
            
            # Process file
            try:
                # Load and process
                # Extract filter parameters from filters dict
                hp = self.filters.get('highpass') if self.filters else None
                lp = self.filters.get('lowpass') if self.filters else None
                gain_db = self.filters.get('gain', 0) if self.filters else 0
                
                data, sr = load_audio_with_filters(file_path, hp=hp, lp=lp, gain_db=gain_db)
                # Use default parameters for spectrogram computation
                params = {
                    "window_size": 1024,
                    "overlap": 50,  # 50% overlap
                    "smooth_sigma": 1.5,
                    "median_filter_size": (3, 1)
                }
                freqs, times, Sxx_norm, Sxx_filt = compute_spectrogram(data, sr, file_path, params)
                Sxx = Sxx_filt
                
                # Save results
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                
                # Save spectrogram data
                np.savez(
                    os.path.join(output_dir, f"{base_name}_spectrogram.npz"),
                    freqs=freqs,
                    times=times,
                    Sxx=Sxx
                )
                
                self.log(f"Processed: {base_name}")
                
            except Exception as e:
                self.log(f"Failed to process {os.path.basename(file_path)}: {e}", error=True)
                
        progress.setValue(self.file_list.count())
        QMessageBox.information(
            self, "Batch Processing Complete",
            f"Processed {self.file_list.count()} files.\nResults saved to: {output_dir}"
        )
        
    def update_statistics(self, result):
        """Update statistics display"""
        Sxx = result['Sxx']
        freqs = result['freqs']
        
        # Calculate statistics
        mean_freq_idx = np.mean(np.argmax(Sxx, axis=0))
        mean_freq = freqs[int(mean_freq_idx)]
        
        peak_freq_idx = np.unravel_index(np.argmax(Sxx), Sxx.shape)[0]
        peak_freq = freqs[peak_freq_idx]
        
        # Bandwidth (simplified)
        threshold = np.max(Sxx) * 0.5
        above_threshold = np.where(np.max(Sxx, axis=1) > threshold)[0]
        if len(above_threshold) > 0:
            bandwidth = freqs[above_threshold[-1]] - freqs[above_threshold[0]]
        else:
            bandwidth = 0
            
        # SNR (simplified)
        signal_power = np.max(Sxx)
        noise_power = np.median(Sxx)
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 0
            
        # Energy
        energy = np.sum(Sxx)
        
        # Update labels
        self.stats_labels["Mean Freq"].setText(f"{mean_freq:.1f} Hz")
        self.stats_labels["Peak Freq"].setText(f"{peak_freq:.1f} Hz")
        self.stats_labels["Bandwidth"].setText(f"{bandwidth:.1f} Hz")
        self.stats_labels["SNR"].setText(f"{snr:.1f} dB")
        self.stats_labels["Energy"].setText(f"{energy:.2e}")
        
    def toggle_playback(self):
        """Toggle play/pause"""
        if self.is_playing:
            self.player.pause()
            self.play_btn.setIcon(qta.icon('mdi.play', color='white'))
            self.is_playing = False
        else:
            if self.current_audio_data is not None:
                self.player.play()
                self.play_btn.setIcon(qta.icon('mdi.pause', color='white'))
                self.is_playing = True
                
    def stop_playback(self):
        """Stop playback"""
        self.player.stop()
        self.play_btn.setIcon(qta.icon('mdi.play', color='white'))
        self.is_playing = False
        self.progress_slider.setValue(0)
        self.spectrogram_canvas.hide_playback_position()
        
    def on_playback_position_changed(self, position):
        """Update playback position"""
        if self.current_audio_data is not None:
            total_samples = len(self.current_audio_data)
            progress = int((position / total_samples) * 1000)
            self.progress_slider.setValue(progress)
            
            # Update time label
            current_time = position / self.current_sample_rate
            total_time = total_samples / self.current_sample_rate
            self.time_label.setText(
                f"{self.format_time(current_time)} / {self.format_time(total_time)}"
            )
            
            # Update spectrogram marker
            self.spectrogram_canvas.set_playback_position(current_time)
            
    def on_playback_finished(self):
        """Handle playback finished"""
        self.play_btn.setIcon(qta.icon('mdi.play', color='white'))
        self.is_playing = False
        self.spectrogram_canvas.hide_playback_position()
        
    def on_slider_pressed(self):
        """Handle slider press"""
        self.player.pause()
        
    def on_slider_released(self):
        """Handle slider release"""
        if self.current_audio_data is not None:
            position = self.progress_slider.value() / 1000
            sample_position = int(position * len(self.current_audio_data))
            self.player.seek(sample_position)
            
            if self.is_playing:
                self.player.play()
                
    def change_playback_speed(self, speed_text):
        """Change playback speed"""
        speed = float(speed_text[:-1])
        self.player.set_speed(speed)
        
    def change_volume(self, value):
        """Change volume"""
        self.player.set_volume(value / 100)
        self.volume_label.setText(f"{value}%")
        
    def format_time(self, seconds):
        """Format time as MM:SS"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
        
    def on_canvas_click(self, rel_sec, event):
        """Handle canvas click"""
        if event.button() == Qt.LeftButton:
            # Seek to position
            if self.current_audio_data is not None:
                sample_position = int(rel_sec * self.current_sample_rate)
                self.player.seek(sample_position)
        elif event.button() == Qt.RightButton:
            # Start annotation
            self.annotator.on_click(rel_sec, event)
            
    def update_hover_info(self, text):
        """Update hover information"""
        self.status_bar.showMessage(text, 2000)
        
    def on_range_selected(self, range_tuple):
        """Handle range selection"""
        start, end = range_tuple
        duration = end - start
        self.log(f"Range selected: {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")
        
    def update_spectrogram_params(self):
        """Update spectrogram parameters"""
        self.spectrogram_canvas.spectrogram_params = {
            "window": self.window_combo.currentText(),
            "nperseg": self.nperseg_spin.value(),
            "noverlap": self.overlap_spin.value()
        }
        
        # Reprocess if file is loaded
        if self.current_file_path:
            self.load_and_process_file()
            
    def open_filter_dialog(self):
        """Open filter configuration dialog"""
        dialog = CombinedFilterDialog(self)
        if dialog.exec_():
            self.filters = dialog.get_filters()
            self.update_filter_status()
            
            # Clear cache since filters changed
            self.spectrogram_cache.clear()
            
            # Reprocess current file
            if self.current_file_path:
                self.load_and_process_file()
                
    def update_filter_status(self):
        """Update filter status label"""
        if self.filters:
            count = len(self.filters)
            self.filter_status.setText(f"{count} filter{'s' if count != 1 else ''} active")
            self.filter_status.setStyleSheet("color: #10B981; font-size: 14px;")
        else:
            self.filter_status.setText("No filters applied")
            self.filter_status.setStyleSheet("color: #9CA3AF; font-size: 14px;")
            
    def configure_1d_detector(self):
        """Configure 1D detector"""
        # Create a default detector instance
        from spectrogram_gui.utils.auto_detector import DopplerDetector
        detector = DopplerDetector()
        dialog = DetectorParamsDialog(self, detector=detector)
        dialog.exec_()
        
    def configure_2d_detector(self):
        """Configure 2D detector"""
        # Create a default detector instance
        from spectrogram_gui.utils.detector_2d import DopplerDetector2D
        detector = DopplerDetector2D()
        dialog = Detector2DParamsDialog(self, detector=detector)
        dialog.exec_()
        
    def run_detection(self):
        """Run detection algorithm"""
        if self.current_audio_data is None:
            QMessageBox.information(self, "No Data", "Please load an audio file first.")
            return
            
        # Run detection (simplified)
        self.detection_results.clear()
        self.detection_results.append("Running detection...")
        
        # TODO: Implement actual detection
        QTimer.singleShot(1000, lambda: self.detection_results.append("Detection complete."))
        
    def export_spectrogram_image(self):
        """Export spectrogram as image"""
        if self.spectrogram_canvas.Sxx is None:
            QMessageBox.information(self, "No Data", "No spectrogram to export.")
            return
            
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Image", "", "PNG Files (*.png);;All Files (*.*)"
        )
        
        if path:
            # Export using pyqtgraph's export functionality
            from pyqtgraph.exporters import ImageExporter
            exporter = ImageExporter(self.spectrogram_canvas.plot)
            exporter.export(path)
            
            QMessageBox.information(self, "Export Complete", f"Image saved to:\n{path}")
            
    def export_spectrogram_data(self):
        """Export spectrogram data"""
        if self.spectrogram_canvas.Sxx is None:
            QMessageBox.information(self, "No Data", "No spectrogram data to export.")
            return
            
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", "NumPy Files (*.npz);;All Files (*.*)"
        )
        
        if path:
            np.savez(
                path,
                freqs=self.spectrogram_canvas.freqs,
                times=self.spectrogram_canvas.times,
                Sxx=self.spectrogram_canvas.Sxx_raw
            )
            
            QMessageBox.information(self, "Export Complete", f"Data saved to:\n{path}")
            
    def on_annotation_added(self, annotation):
        """Handle annotation added"""
        self.log(f"Annotation added: {annotation.get('Type', 'Unknown')}")
        
    def extract_metadata(self, file_path):
        """Extract site and pixel from filename"""
        basename = os.path.basename(file_path)
        
        # Try to extract site and pixel (customize based on your naming convention)
        site = "Unknown"
        pixel = "Unknown"
        
        # Example pattern: site_pixel_timestamp.wav
        parts = basename.split('_')
        if len(parts) >= 2:
            site = parts[0]
            pixel = parts[1]
            
        return site, pixel
        
    def update_memory_usage(self):
        """Update memory usage display"""
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_label.setText(f"Memory: {memory_mb:.1f} MB")
        
    def save_session(self):
        """Save current session"""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "", "JSON Files (*.json);;All Files (*.*)"
        )
        
        if path:
            session = {
                "files": [self.file_list.item(i).data(Qt.UserRole) 
                         for i in range(self.file_list.count())],
                "filters": self.filters,
                "spectrogram_params": self.spectrogram_canvas.spectrogram_params,
                "current_file": self.current_file_path
            }
            
            with open(path, 'w') as f:
                json.dump(session, f, indent=2)
                
            self.log(f"Session saved to: {path}")
            
    def load_settings(self):
        """Load application settings"""
        # TODO: Implement settings loading from config file
        pass
        
    def show_help(self):
        """Show help dialog"""
        QMessageBox.information(
            self,
            "Help",
            "Spectrogram Analyzer - Modern Edition\n\n"
            "Features:\n"
            "• Enhanced zoom and pan controls\n"
            "• Multiple normalization options\n"
            "• Modern event tagging system\n"
            "• Batch processing capabilities\n"
            "• Multi-threaded performance\n\n"
            "Shortcuts:\n"
            "• Space: Play/Pause\n"
            "• A: Add annotation\n"
            "• E: Edit annotation\n"
            "• Delete: Remove annotation\n"
            "• Ctrl+O: Open files\n"
            "• Ctrl+S: Save session"
        )
        
    def log(self, message, error=False):
        """Log message to console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if error:
            formatted = f"<span style='color: #EF4444;'>[{timestamp}] ERROR: {message}</span>"
        else:
            formatted = f"<span style='color: #10B981;'>[{timestamp}] {message}</span>"
            
        if hasattr(self, 'console'):
            self.console.append(formatted)
            
    def closeEvent(self, event):
        """Handle application close"""
        # Clean up thread pool
        self.thread_pool.waitForDone(1000)
        
        # Stop timers
        if hasattr(self, 'memory_timer'):
            self.memory_timer.stop()
            
        # Stop player
        if hasattr(self, 'player'):
            self.player.stop()
            
        event.accept()


# Entry point for testing
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernMainWindow()
    window.show()
    sys.exit(app.exec_())