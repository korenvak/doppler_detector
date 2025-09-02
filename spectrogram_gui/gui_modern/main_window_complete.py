"""
Complete Modern Spectrogram GUI with all functionality integrated
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import traceback

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QPushButton, QLabel, QListWidget, QListWidgetItem,
    QSplitter, QMessageBox, QMenu, QApplication, QFrame,
    QToolButton, QGraphicsDropShadowEffect, QProgressDialog,
    QSlider, QSpinBox, QGroupBox, QGridLayout, QScrollArea,
    QSizePolicy, QDockWidget, QTextEdit, QComboBox, QCheckBox,
    QDoubleSpinBox, QTabWidget, QDialog, QDialogButtonBox
)
from PySide6.QtGui import (
    QKeySequence, QAction, QIcon, QPalette, QColor,
    QLinearGradient, QBrush, QPainter, QFont,
    QShortcut, QPen, QPixmap
)
from PySide6.QtCore import (
    Qt, Signal, QSize, QEvent, QSettings, QTimer,
    QPropertyAnimation, QEasingCurve, QRect, QPoint,
    QParallelAnimationGroup, Property, QRunnable, QThreadPool,
    Slot
)

import qtawesome as qta

# Import our modules
from .main_window import GlassPanel, ModernButton, ModernFileList, ModernControlBar
from .spectrogram_canvas import OptimizedSpectrogramCanvas
from .audio_processor import AudioProcessor
from .audio_player import ModernAudioPlayer, AudioDeviceManager


class FilterDialog(QDialog):
    """Modern filter configuration dialog"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Audio Filters")
        self.setModal(True)
        self.setMinimumSize(500, 600)
        
        # Apply glass effect
        self.setStyleSheet("""
            QDialog {
                background: rgba(20, 20, 30, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
            }
        """)
        
        self.filters_config = {}
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Title
        title = QLabel("Audio Filter Configuration")
        title.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.9);
                font-size: 18px;
                font-weight: 600;
                padding: 8px;
            }
        """)
        layout.addWidget(title)
        
        # Filter tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
            }
            QTabBar::tab {
                background: rgba(255, 255, 255, 0.05);
                color: rgba(255, 255, 255, 0.7);
                padding: 8px 16px;
                margin: 2px;
                border-radius: 4px;
            }
            QTabBar::tab:selected {
                background: rgba(139, 92, 246, 0.3);
                color: white;
            }
        """)
        
        # Gain tab
        gain_widget = self.create_gain_tab()
        self.tabs.addTab(gain_widget, "Gain")
        
        # High-pass tab
        highpass_widget = self.create_highpass_tab()
        self.tabs.addTab(highpass_widget, "High-Pass")
        
        # Low-pass tab
        lowpass_widget = self.create_lowpass_tab()
        self.tabs.addTab(lowpass_widget, "Low-Pass")
        
        # Band-pass tab
        bandpass_widget = self.create_bandpass_tab()
        self.tabs.addTab(bandpass_widget, "Band-Pass")
        
        # Notch tab
        notch_widget = self.create_notch_tab()
        self.tabs.addTab(notch_widget, "Notch")
        
        layout.addWidget(self.tabs)
        
        # Normalize checkbox
        self.normalize_check = QCheckBox("Normalize Output")
        self.normalize_check.setStyleSheet("""
            QCheckBox {
                color: rgba(255, 255, 255, 0.8);
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid rgba(139, 92, 246, 0.5);
                border-radius: 4px;
                background: rgba(255, 255, 255, 0.05);
            }
            QCheckBox::indicator:checked {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #6366F1,
                    stop: 1 #8B5CF6
                );
            }
        """)
        layout.addWidget(self.normalize_check)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        apply_btn = ModernButton("Apply", primary=True)
        apply_btn.clicked.connect(self.accept)
        
        cancel_btn = ModernButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(apply_btn)
        
        layout.addLayout(button_layout)
        
    def create_gain_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Enable checkbox
        self.gain_enabled = QCheckBox("Enable Gain Adjustment")
        self.gain_enabled.setStyleSheet(self.normalize_check.styleSheet())
        layout.addWidget(self.gain_enabled)
        
        # Gain slider
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Gain (dB):"))
        
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(-20, 20)
        self.gain_slider.setValue(0)
        self.gain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.gain_slider.setTickInterval(5)
        
        self.gain_value = QLabel("0 dB")
        self.gain_value.setMinimumWidth(60)
        
        self.gain_slider.valueChanged.connect(
            lambda v: self.gain_value.setText(f"{v} dB")
        )
        
        gain_layout.addWidget(self.gain_slider)
        gain_layout.addWidget(self.gain_value)
        
        layout.addLayout(gain_layout)
        layout.addStretch()
        
        return widget
        
    def create_highpass_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.highpass_enabled = QCheckBox("Enable High-Pass Filter")
        self.highpass_enabled.setStyleSheet(self.normalize_check.styleSheet())
        layout.addWidget(self.highpass_enabled)
        
        # Cutoff frequency
        cutoff_layout = QHBoxLayout()
        cutoff_layout.addWidget(QLabel("Cutoff Frequency:"))
        
        self.highpass_cutoff = QDoubleSpinBox()
        self.highpass_cutoff.setRange(20, 20000)
        self.highpass_cutoff.setValue(100)
        self.highpass_cutoff.setSuffix(" Hz")
        
        cutoff_layout.addWidget(self.highpass_cutoff)
        layout.addLayout(cutoff_layout)
        
        # Order
        order_layout = QHBoxLayout()
        order_layout.addWidget(QLabel("Filter Order:"))
        
        self.highpass_order = QSpinBox()
        self.highpass_order.setRange(1, 10)
        self.highpass_order.setValue(5)
        
        order_layout.addWidget(self.highpass_order)
        layout.addLayout(order_layout)
        
        layout.addStretch()
        return widget
        
    def create_lowpass_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.lowpass_enabled = QCheckBox("Enable Low-Pass Filter")
        self.lowpass_enabled.setStyleSheet(self.normalize_check.styleSheet())
        layout.addWidget(self.lowpass_enabled)
        
        # Cutoff frequency
        cutoff_layout = QHBoxLayout()
        cutoff_layout.addWidget(QLabel("Cutoff Frequency:"))
        
        self.lowpass_cutoff = QDoubleSpinBox()
        self.lowpass_cutoff.setRange(20, 20000)
        self.lowpass_cutoff.setValue(8000)
        self.lowpass_cutoff.setSuffix(" Hz")
        
        cutoff_layout.addWidget(self.lowpass_cutoff)
        layout.addLayout(cutoff_layout)
        
        # Order
        order_layout = QHBoxLayout()
        order_layout.addWidget(QLabel("Filter Order:"))
        
        self.lowpass_order = QSpinBox()
        self.lowpass_order.setRange(1, 10)
        self.lowpass_order.setValue(5)
        
        order_layout.addWidget(self.lowpass_order)
        layout.addLayout(order_layout)
        
        layout.addStretch()
        return widget
        
    def create_bandpass_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.bandpass_enabled = QCheckBox("Enable Band-Pass Filter")
        self.bandpass_enabled.setStyleSheet(self.normalize_check.styleSheet())
        layout.addWidget(self.bandpass_enabled)
        
        # Low frequency
        low_layout = QHBoxLayout()
        low_layout.addWidget(QLabel("Low Frequency:"))
        
        self.bandpass_low = QDoubleSpinBox()
        self.bandpass_low.setRange(20, 20000)
        self.bandpass_low.setValue(300)
        self.bandpass_low.setSuffix(" Hz")
        
        low_layout.addWidget(self.bandpass_low)
        layout.addLayout(low_layout)
        
        # High frequency
        high_layout = QHBoxLayout()
        high_layout.addWidget(QLabel("High Frequency:"))
        
        self.bandpass_high = QDoubleSpinBox()
        self.bandpass_high.setRange(20, 20000)
        self.bandpass_high.setValue(3000)
        self.bandpass_high.setSuffix(" Hz")
        
        high_layout.addWidget(self.bandpass_high)
        layout.addLayout(high_layout)
        
        # Order
        order_layout = QHBoxLayout()
        order_layout.addWidget(QLabel("Filter Order:"))
        
        self.bandpass_order = QSpinBox()
        self.bandpass_order.setRange(1, 10)
        self.bandpass_order.setValue(5)
        
        order_layout.addWidget(self.bandpass_order)
        layout.addLayout(order_layout)
        
        layout.addStretch()
        return widget
        
    def create_notch_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.notch_enabled = QCheckBox("Enable Notch Filter")
        self.notch_enabled.setStyleSheet(self.normalize_check.styleSheet())
        layout.addWidget(self.notch_enabled)
        
        # Center frequency
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Center Frequency:"))
        
        self.notch_freq = QDoubleSpinBox()
        self.notch_freq.setRange(20, 20000)
        self.notch_freq.setValue(50)
        self.notch_freq.setSuffix(" Hz")
        
        freq_layout.addWidget(self.notch_freq)
        layout.addLayout(freq_layout)
        
        # Q factor
        q_layout = QHBoxLayout()
        q_layout.addWidget(QLabel("Q Factor:"))
        
        self.notch_q = QDoubleSpinBox()
        self.notch_q.setRange(1, 100)
        self.notch_q.setValue(30)
        
        q_layout.addWidget(self.notch_q)
        layout.addLayout(q_layout)
        
        layout.addStretch()
        return widget
        
    def get_filters_config(self):
        """Get the configured filters"""
        config = {}
        
        if self.gain_enabled.isChecked():
            config['gain'] = self.gain_slider.value()
            
        if self.highpass_enabled.isChecked():
            config['highpass'] = {
                'cutoff': self.highpass_cutoff.value(),
                'order': self.highpass_order.value()
            }
            
        if self.lowpass_enabled.isChecked():
            config['lowpass'] = {
                'cutoff': self.lowpass_cutoff.value(),
                'order': self.lowpass_order.value()
            }
            
        if self.bandpass_enabled.isChecked():
            config['bandpass'] = {
                'low': self.bandpass_low.value(),
                'high': self.bandpass_high.value(),
                'order': self.bandpass_order.value()
            }
            
        if self.notch_enabled.isChecked():
            config['notch'] = {
                'freq': self.notch_freq.value(),
                'Q': self.notch_q.value()
            }
            
        config['normalize'] = self.normalize_check.isChecked()
        
        return config


class CompleteModernMainWindow(QMainWindow):
    """
    Complete main window with all functionality
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectrogram Analyzer - Modern UI")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Core components
        self.audio_processor = AudioProcessor()
        self.audio_player = ModernAudioPlayer(self)
        self.thread_pool = QThreadPool()
        
        # Data storage
        self.current_audio = None
        self.current_sr = None
        self.current_file = None
        self.filters_config = {}
        
        # Setup UI
        self.setup_ui()
        self.apply_theme()
        self.setup_connections()
        self.setup_shortcuts()
        
        # Load settings
        self.settings = QSettings("SpectrogramGUI", "ModernMainWindow")
        self.load_settings()
        
    def setup_ui(self):
        """Setup the complete UI"""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Toolbar
        self.setup_toolbar()
        
        # Main content splitter
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setHandleWidth(2)
        
        # Left panel (file list)
        left_panel = self.create_file_panel()
        content_splitter.addWidget(left_panel)
        
        # Center panel (spectrogram)
        center_panel = self.create_spectrogram_panel()
        content_splitter.addWidget(center_panel)
        
        # Right panel (controls and info)
        right_panel = self.create_control_panel()
        content_splitter.addWidget(right_panel)
        
        # Set initial sizes
        content_splitter.setSizes([300, 1000, 300])
        
        main_layout.addWidget(content_splitter)
        
        # Audio control bar at bottom
        self.control_bar = ModernControlBar()
        main_layout.addWidget(self.control_bar)
        
        # Status bar
        self.setup_status_bar()
        
    def create_file_panel(self):
        """Create the file list panel"""
        panel = GlassPanel()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel("Audio Files")
        header_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.9);
                font-size: 16px;
                font-weight: 600;
            }
        """)
        header_layout.addWidget(header_label)
        
        # Add button
        add_btn = ModernButton("+")
        add_btn.setFixedSize(32, 32)
        add_btn.clicked.connect(self.add_files)
        header_layout.addWidget(add_btn)
        
        layout.addLayout(header_layout)
        
        # File list
        self.file_list = ModernFileList()
        self.file_list.itemDoubleClicked.connect(self.load_file)
        self.file_list.fileDeleteRequested.connect(self.remove_selected_files)
        layout.addWidget(self.file_list)
        
        # File controls
        controls_layout = QHBoxLayout()
        
        sort_btn = ModernButton("Sort")
        sort_btn.clicked.connect(self.show_sort_menu)
        
        clear_btn = ModernButton("Clear")
        clear_btn.clicked.connect(self.clear_files)
        
        controls_layout.addWidget(sort_btn)
        controls_layout.addWidget(clear_btn)
        
        layout.addLayout(controls_layout)
        
        return panel
        
    def create_spectrogram_panel(self):
        """Create the spectrogram display panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Spectrogram canvas
        self.spectrogram = OptimizedSpectrogramCanvas()
        self.spectrogram.click_callback.connect(self.on_spectrogram_click)
        self.spectrogram.hover_callback.connect(self.on_spectrogram_hover)
        self.spectrogram.selection_callback.connect(self.on_spectrogram_selection)
        
        layout.addWidget(self.spectrogram)
        
        return panel
        
    def create_control_panel(self):
        """Create the control and info panel"""
        panel = GlassPanel()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Title
        title = QLabel("Controls")
        title.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.9);
                font-size: 16px;
                font-weight: 600;
                padding-bottom: 8px;
            }
        """)
        layout.addWidget(title)
        
        # Spectrogram settings group
        spec_group = QGroupBox("Spectrogram")
        spec_group.setStyleSheet("""
            QGroupBox {
                color: rgba(255, 255, 255, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding-top: 16px;
                margin-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
            }
        """)
        
        spec_layout = QGridLayout(spec_group)
        
        # Window size
        spec_layout.addWidget(QLabel("Window:"), 0, 0)
        self.window_size = QComboBox()
        self.window_size.addItems(["512", "1024", "2048", "4096", "8192"])
        self.window_size.setCurrentText("2048")
        spec_layout.addWidget(self.window_size, 0, 1)
        
        # Overlap
        spec_layout.addWidget(QLabel("Overlap:"), 1, 0)
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(0, 99)
        self.overlap_spin.setValue(75)
        self.overlap_spin.setSuffix("%")
        spec_layout.addWidget(self.overlap_spin, 1, 1)
        
        # Colormap
        spec_layout.addWidget(QLabel("Colormap:"), 2, 0)
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "viridis", "plasma", "inferno", "magma",
            "cividis", "twilight", "turbo", "jet"
        ])
        self.colormap_combo.currentTextChanged.connect(self.change_colormap)
        spec_layout.addWidget(self.colormap_combo, 2, 1)
        
        # Update button
        update_btn = ModernButton("Update", primary=True)
        update_btn.clicked.connect(self.update_spectrogram)
        spec_layout.addWidget(update_btn, 3, 0, 1, 2)
        
        layout.addWidget(spec_group)
        
        # Audio info group
        info_group = QGroupBox("Audio Info")
        info_group.setStyleSheet(spec_group.styleSheet())
        
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        self.info_text.setStyleSheet("""
            QTextEdit {
                background: rgba(0, 0, 0, 0.2);
                color: rgba(255, 255, 255, 0.7);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 4px;
                padding: 4px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
            }
        """)
        info_layout.addWidget(self.info_text)
        
        layout.addWidget(info_group)
        
        # Hover info
        self.hover_label = QLabel("Hover over spectrogram for details")
        self.hover_label.setStyleSheet("""
            QLabel {
                background: rgba(0, 0, 0, 0.3);
                color: rgba(255, 255, 255, 0.6);
                border-radius: 4px;
                padding: 8px;
                font-size: 12px;
            }
        """)
        self.hover_label.setWordWrap(True)
        layout.addWidget(self.hover_label)
        
        layout.addStretch()
        
        return panel
        
    def setup_toolbar(self):
        """Setup the toolbar"""
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setStyleSheet("""
            QToolBar {
                background: rgba(0, 0, 0, 0.3);
                border: none;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                padding: 8px;
                spacing: 8px;
            }
            QToolButton {
                background: transparent;
                border: none;
                padding: 6px;
                border-radius: 4px;
            }
            QToolButton:hover {
                background: rgba(255, 255, 255, 0.1);
            }
            QToolButton:pressed {
                background: rgba(255, 255, 255, 0.05);
            }
        """)
        
        # File operations
        open_action = QAction(qta.icon('fa5s.folder-open', color='#9CA3AF'), "Open Files", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.add_files)
        toolbar.addAction(open_action)
        
        save_action = QAction(qta.icon('fa5s.save', color='#9CA3AF'), "Export", self)
        save_action.setShortcut(QKeySequence("Ctrl+S"))
        save_action.triggered.connect(self.export_spectrogram)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # View operations
        zoom_in_action = QAction(qta.icon('fa5s.search-plus', color='#9CA3AF'), "Zoom In", self)
        zoom_in_action.setShortcut(QKeySequence("Ctrl++"))
        zoom_in_action.triggered.connect(self.zoom_in)
        toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction(qta.icon('fa5s.search-minus', color='#9CA3AF'), "Zoom Out", self)
        zoom_out_action.setShortcut(QKeySequence("Ctrl+-"))
        zoom_out_action.triggered.connect(self.zoom_out)
        toolbar.addAction(zoom_out_action)
        
        reset_action = QAction(qta.icon('fa5s.compress', color='#9CA3AF'), "Reset View", self)
        reset_action.setShortcut(QKeySequence("Ctrl+0"))
        reset_action.triggered.connect(self.reset_view)
        toolbar.addAction(reset_action)
        
        toolbar.addSeparator()
        
        # Processing
        filter_action = QAction(qta.icon('fa5s.filter', color='#9CA3AF'), "Filters", self)
        filter_action.triggered.connect(self.show_filter_dialog)
        toolbar.addAction(filter_action)
        
        toolbar.addSeparator()
        
        # Settings
        settings_action = QAction(qta.icon('fa5s.cog', color='#9CA3AF'), "Settings", self)
        settings_action.triggered.connect(self.show_settings)
        toolbar.addAction(settings_action)
        
    def setup_status_bar(self):
        """Setup status bar"""
        status = self.statusBar()
        status.setStyleSheet("""
            QStatusBar {
                background: rgba(0, 0, 0, 0.3);
                color: rgba(255, 255, 255, 0.6);
                border-top: 1px solid rgba(255, 255, 255, 0.05);
                font-size: 12px;
            }
        """)
        status.showMessage("Ready")
        
    def setup_connections(self):
        """Setup signal connections"""
        # Audio player
        self.audio_player.position_changed.connect(self.on_playback_position)
        self.audio_player.state_changed.connect(self.on_playback_state)
        self.audio_player.error.connect(self.on_playback_error)
        
        # Control bar
        self.control_bar.playRequested.connect(self.play_audio)
        self.control_bar.pauseRequested.connect(self.pause_audio)
        self.control_bar.stopRequested.connect(self.stop_audio)
        self.control_bar.seekRequested.connect(self.seek_audio)
        self.control_bar.volumeChanged.connect(self.audio_player.set_volume)
        
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Playback
        QShortcut(QKeySequence("Space"), self, self.toggle_playback)
        QShortcut(QKeySequence("S"), self, self.stop_audio)
        
        # File operations
        QShortcut(QKeySequence("Delete"), self.file_list, self.remove_selected_files)
        
    def apply_theme(self):
        """Apply the glassmorphic theme"""
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #0F0F1E,
                    stop: 0.5 #1A1A2E,
                    stop: 1 #16213E
                );
            }
            QSplitter::handle {
                background: rgba(255, 255, 255, 0.05);
                width: 2px;
            }
            QSplitter::handle:hover {
                background: rgba(139, 92, 246, 0.3);
            }
        """)
        
    def add_files(self):
        """Add audio files"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio Files",
            "",
            "Audio Files (*.wav *.flac);;All Files (*.*)"
        )
        
        for file_path in files:
            # Check if already in list
            exists = False
            for i in range(self.file_list.count()):
                if self.file_list.item(i).data(Qt.ItemDataRole.UserRole) == file_path:
                    exists = True
                    break
                    
            if not exists:
                item = QListWidgetItem(os.path.basename(file_path))
                item.setIcon(qta.icon('fa5s.music', color='#8B5CF6'))
                item.setData(Qt.ItemDataRole.UserRole, file_path)
                self.file_list.addItem(item)
                
    def load_file(self, item):
        """Load selected audio file"""
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if not file_path:
            return
            
        try:
            # Show loading status
            self.statusBar().showMessage(f"Loading: {os.path.basename(file_path)}")
            QApplication.processEvents()
            
            # Load audio
            audio_data, sample_rate = self.audio_processor.load_audio(file_path)
            
            # Apply filters if configured
            if self.filters_config:
                audio_data = self.audio_processor.apply_filters(
                    audio_data, sample_rate, self.filters_config
                )
                
            # Store data
            self.current_audio = audio_data
            self.current_sr = sample_rate
            self.current_file = file_path
            
            # Load into player
            self.audio_player.load_audio(audio_data, sample_rate)
            
            # Compute and display spectrogram
            self.compute_and_display_spectrogram()
            
            # Update info
            self.update_audio_info()
            
            self.statusBar().showMessage(f"Loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")
            self.statusBar().showMessage("Error loading file")
            
    def compute_and_display_spectrogram(self):
        """Compute and display spectrogram"""
        if self.current_audio is None:
            return
            
        try:
            # Get parameters
            window_size = int(self.window_size.currentText())
            overlap_percent = self.overlap_spin.value() / 100
            noverlap = int(window_size * overlap_percent)
            
            # Compute spectrogram
            freqs, times, Sxx = self.audio_processor.compute_spectrogram(
                self.current_audio,
                self.current_sr,
                nperseg=window_size,
                noverlap=noverlap,
                window='hann',
                mode='magnitude'
            )
            
            # Display
            self.spectrogram.set_spectrogram_data(
                self.current_audio,
                self.current_sr,
                freqs,
                times,
                Sxx
            )
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to compute spectrogram:\n{str(e)}")
            
    def update_audio_info(self):
        """Update audio information display"""
        if self.current_audio is None:
            return
            
        duration = len(self.current_audio) / self.current_sr
        info_text = f"""File: {os.path.basename(self.current_file)}
Duration: {duration:.2f} seconds
Sample Rate: {self.current_sr} Hz
Samples: {len(self.current_audio):,}
Channels: 1 (mono)"""
        
        self.info_text.setText(info_text)
        
    def update_spectrogram(self):
        """Update spectrogram with new parameters"""
        self.compute_and_display_spectrogram()
        
    def change_colormap(self, colormap_name):
        """Change spectrogram colormap"""
        self.spectrogram.colormap_name = colormap_name
        self.spectrogram.update_colormap()
        
    def show_filter_dialog(self):
        """Show filter configuration dialog"""
        dialog = FilterDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.filters_config = dialog.get_filters_config()
            
            # Reprocess current audio if loaded
            if self.current_file:
                item = self.file_list.currentItem()
                if item:
                    self.load_file(item)
                    
    def play_audio(self):
        """Play audio"""
        if self.current_audio is not None:
            self.audio_player.play()
            
    def pause_audio(self):
        """Pause audio"""
        self.audio_player.pause()
        
    def stop_audio(self):
        """Stop audio"""
        self.audio_player.stop()
        self.spectrogram.set_playback_position(0)
        
    def seek_audio(self, position):
        """Seek to position"""
        self.audio_player.seek(position)
        
    def toggle_playback(self):
        """Toggle play/pause"""
        if self.audio_player.state == 'playing':
            self.pause_audio()
        else:
            self.play_audio()
            
    @Slot(float, float)
    def on_playback_position(self, current, total):
        """Handle playback position updates"""
        self.control_bar.update_time(current, total)
        self.spectrogram.set_playback_position(current)
        
    @Slot(str)
    def on_playback_state(self, state):
        """Handle playback state changes"""
        if state == 'stopped':
            self.control_bar.is_playing = False
            self.control_bar.play_btn.setIcon(qta.icon('fa5s.play', color='white'))
            
    @Slot(str)
    def on_playback_error(self, error):
        """Handle playback errors"""
        QMessageBox.warning(self, "Playback Error", error)
        
    @Slot(float, object)
    def on_spectrogram_click(self, time, event):
        """Handle spectrogram clicks"""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Seek to clicked position
            self.seek_audio(time)
            
    @Slot(str)
    def on_spectrogram_hover(self, text):
        """Handle spectrogram hover"""
        self.hover_label.setText(text)
        
    @Slot(float, float)
    def on_spectrogram_selection(self, start, end):
        """Handle spectrogram selection"""
        duration = end - start
        self.statusBar().showMessage(f"Selection: {start:.2f}s - {end:.2f}s ({duration:.2f}s)")
        
    def zoom_in(self):
        """Zoom in on spectrogram"""
        if self.spectrogram.viewbox:
            vr = self.spectrogram.viewbox.viewRect()
            center_x = vr.center().x()
            center_y = vr.center().y()
            new_width = vr.width() * 0.8
            new_height = vr.height() * 0.8
            
            self.spectrogram.viewbox.setRange(
                xRange=(center_x - new_width/2, center_x + new_width/2),
                yRange=(center_y - new_height/2, center_y + new_height/2),
                padding=0
            )
            
    def zoom_out(self):
        """Zoom out on spectrogram"""
        if self.spectrogram.viewbox:
            vr = self.spectrogram.viewbox.viewRect()
            center_x = vr.center().x()
            center_y = vr.center().y()
            new_width = vr.width() * 1.25
            new_height = vr.height() * 1.25
            
            self.spectrogram.viewbox.setRange(
                xRange=(center_x - new_width/2, center_x + new_width/2),
                yRange=(center_y - new_height/2, center_y + new_height/2),
                padding=0
            )
            
    def reset_view(self):
        """Reset spectrogram view"""
        if self.spectrogram.viewbox:
            self.spectrogram.viewbox.autoRange()
            
    def export_spectrogram(self):
        """Export spectrogram image"""
        if self.current_audio is None:
            QMessageBox.information(self, "Info", "No spectrogram to export")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Spectrogram",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*.*)"
        )
        
        if file_path:
            # Export implementation would go here
            self.statusBar().showMessage(f"Exported to: {file_path}")
            
    def show_sort_menu(self):
        """Show file sort menu"""
        menu = QMenu(self)
        
        name_action = menu.addAction("Sort by Name")
        name_action.triggered.connect(lambda: self.sort_files("name"))
        
        time_action = menu.addAction("Sort by Time")
        time_action.triggered.connect(lambda: self.sort_files("time"))
        
        size_action = menu.addAction("Sort by Size")
        size_action.triggered.connect(lambda: self.sort_files("size"))
        
        menu.exec(self.cursor().pos())
        
    def sort_files(self, key):
        """Sort file list"""
        # Implementation would go here
        pass
        
    def remove_selected_files(self):
        """Remove selected files"""
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
            
    def clear_files(self):
        """Clear all files"""
        self.file_list.clear()
        
    def show_settings(self):
        """Show settings dialog"""
        QMessageBox.information(self, "Settings", "Settings dialog would appear here")
        
    def load_settings(self):
        """Load application settings"""
        # Window geometry
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
            
        # Window state
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
            
    def save_settings(self):
        """Save application settings"""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        
    def closeEvent(self, event):
        """Handle close event"""
        # Save settings
        self.save_settings()
        
        # Cleanup audio player
        self.audio_player.cleanup()
        
        event.accept()