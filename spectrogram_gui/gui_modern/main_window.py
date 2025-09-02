"""
Modern Spectrogram GUI with Glassmorphic Design
Built with PySide6 for better performance and modern UI
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from functools import lru_cache

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QPushButton, QLabel, QListWidget, QListWidgetItem,
    QSplitter, QMessageBox, QMenu, QApplication, QFrame,
    QToolButton, QGraphicsDropShadowEffect, QProgressDialog,
    QSlider, QSpinBox, QGroupBox, QGridLayout, QScrollArea,
    QSizePolicy, QGraphicsBlurEffect, QGraphicsOpacityEffect
)
from PySide6.QtGui import (
    QKeySequence, QAction, QIcon, QPalette, QColor, 
    QLinearGradient, QBrush, QPainter, QFont, QFontDatabase,
    QShortcut, QPen, QPixmap
)
from PySide6.QtCore import (
    Qt, Signal, QSize, QEvent, QSettings, QTimer,
    QPropertyAnimation, QEasingCurve, QRect, QPoint,
    QParallelAnimationGroup, QSequentialAnimationGroup,
    Property, QRunnable, QThreadPool
)

import qtawesome as qta


class GlassPanel(QFrame):
    """
    Glass morphism panel with blur effect and subtle borders
    """
    def __init__(self, parent=None, blur_radius=16):
        super().__init__(parent)
        self.setObjectName("GlassPanel")
        self.blur_radius = blur_radius
        self.setup_glass_effect()
        
    def setup_glass_effect(self):
        """Apply glass morphism effect"""
        self.setStyleSheet("""
            QFrame#GlassPanel {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(255, 255, 255, 0.08),
                    stop: 1 rgba(255, 255, 255, 0.03)
                );
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
            }
        """)
        
        # Add drop shadow for depth
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 60))
        self.setGraphicsEffect(shadow)


class ModernButton(QPushButton):
    """
    Pill-shaped button with gradient and hover effects
    """
    def __init__(self, text="", parent=None, primary=False):
        super().__init__(text, parent)
        self.primary = primary
        self.setup_style()
        self.setup_animations()
        
    def setup_style(self):
        """Apply modern button styling"""
        if self.primary:
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(
                        x1: 0, y1: 0, x2: 1, y2: 0,
                        stop: 0 #6366F1,
                        stop: 1 #8B5CF6
                    );
                    color: white;
                    border: none;
                    border-radius: 20px;
                    padding: 10px 24px;
                    font-weight: 600;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background: qlineargradient(
                        x1: 0, y1: 0, x2: 1, y2: 0,
                        stop: 0 #7C7FFF,
                        stop: 1 #9F6FFF
                    );
                }
                QPushButton:pressed {
                    background: qlineargradient(
                        x1: 0, y1: 0, x2: 1, y2: 0,
                        stop: 0 #5558E3,
                        stop: 1 #7D4EE8
                    );
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background: rgba(255, 255, 255, 0.05);
                    color: rgba(255, 255, 255, 0.9);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 20px;
                    padding: 10px 24px;
                    font-weight: 500;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }
                QPushButton:pressed {
                    background: rgba(255, 255, 255, 0.03);
                }
            """)
            
    def setup_animations(self):
        """Setup hover animations"""
        self.installEventFilter(self)
        
    def eventFilter(self, obj, event):
        if obj == self:
            if event.type() == QEvent.Type.Enter:
                self.animate_hover(True)
            elif event.type() == QEvent.Type.Leave:
                self.animate_hover(False)
        return super().eventFilter(obj, event)
        
    def animate_hover(self, hover):
        """Animate button on hover"""
        # Add subtle scale animation
        pass  # Will implement with QPropertyAnimation later


class ModernFileList(QListWidget):
    """
    Modern file list with glass morphism and smooth animations
    """
    fileDeleteRequested = Signal()
    filesDropped = Signal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setSpacing(4)
        
        # Modern styling
        self.setStyleSheet("""
            QListWidget {
                background: rgba(255, 255, 255, 0.02);
                border: none;
                border-radius: 12px;
                padding: 8px;
                outline: none;
            }
            QListWidget::item {
                background: rgba(255, 255, 255, 0.05);
                color: rgba(255, 255, 255, 0.9);
                border-radius: 8px;
                padding: 12px;
                margin: 2px 0;
            }
            QListWidget::item:selected {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 rgba(99, 102, 241, 0.3),
                    stop: 1 rgba(139, 92, 246, 0.3)
                );
                border: 1px solid rgba(139, 92, 246, 0.5);
            }
            QListWidget::item:hover {
                background: rgba(255, 255, 255, 0.08);
            }
        """)
        
        # Sorting state
        self.sort_key = "name"
        self.sort_ascending = True
        self.settings = QSettings("SpectrogramGUI", "ModernFileList")
        self.load_sort_settings()
        
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
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
            files = []
            for url in event.mimeData().urls():
                local_path = url.toLocalFile()
                if os.path.isfile(local_path) and local_path.lower().endswith((".wav", ".flac")):
                    if not any(self.item(i).data(Qt.ItemDataRole.UserRole) == local_path
                              for i in range(self.count())):
                        files.append(local_path)
                        item = QListWidgetItem(os.path.basename(local_path))
                        item.setIcon(qta.icon('fa5s.music', color='#8B5CF6'))
                        item.setData(Qt.ItemDataRole.UserRole, local_path)
                        self.addItem(item)
            if files:
                self.filesDropped.emit(files)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)
            
    def load_sort_settings(self):
        """Load sorting preferences"""
        self.sort_key = self.settings.value("sort_key", "name")
        self.sort_ascending = self.settings.value("sort_ascending", True, type=bool)
        
    def save_sort_settings(self):
        """Save sorting preferences"""
        self.settings.setValue("sort_key", self.sort_key)
        self.settings.setValue("sort_ascending", self.sort_ascending)


class ModernControlBar(QWidget):
    """
    Modern audio control bar with glass effect
    """
    playRequested = Signal()
    pauseRequested = Signal()
    stopRequested = Signal()
    seekRequested = Signal(float)
    volumeChanged = Signal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)
        
        # Play/Pause button
        self.play_btn = ModernButton("", primary=True)
        self.play_btn.setIcon(qta.icon('fa5s.play', color='white'))
        self.play_btn.setFixedSize(48, 48)
        self.play_btn.clicked.connect(self.toggle_play)
        layout.addWidget(self.play_btn)
        
        # Stop button
        self.stop_btn = ModernButton("")
        self.stop_btn.setIcon(qta.icon('fa5s.stop', color='#E5E7EB'))
        self.stop_btn.setFixedSize(48, 48)
        self.stop_btn.clicked.connect(self.stopRequested.emit)
        layout.addWidget(self.stop_btn)
        
        # Time display
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.8);
                font-family: 'SF Mono', 'Consolas', monospace;
                font-size: 14px;
                background: rgba(0, 0, 0, 0.2);
                border-radius: 8px;
                padding: 6px 12px;
            }
        """)
        layout.addWidget(self.time_label)
        
        # Progress slider
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                height: 16px;
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #8B5CF6,
                    stop: 1 #6366F1
                );
                border-radius: 8px;
                margin: -5px 0;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #6366F1,
                    stop: 1 #8B5CF6
                );
                border-radius: 3px;
            }
        """)
        self.progress_slider.valueChanged.connect(self.on_seek)
        layout.addWidget(self.progress_slider, 1)
        
        # Volume control
        volume_icon = QLabel()
        volume_icon.setPixmap(qta.icon('fa5s.volume-up', color='#9CA3AF').pixmap(20, 20))
        layout.addWidget(volume_icon)
        
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(70)
        self.volume_slider.setFixedWidth(100)
        self.volume_slider.setStyleSheet(self.progress_slider.styleSheet())
        self.volume_slider.valueChanged.connect(lambda v: self.volumeChanged.emit(v / 100))
        layout.addWidget(self.volume_slider)
        
        self.is_playing = False
        
    def toggle_play(self):
        if self.is_playing:
            self.play_btn.setIcon(qta.icon('fa5s.play', color='white'))
            self.pauseRequested.emit()
        else:
            self.play_btn.setIcon(qta.icon('fa5s.pause', color='white'))
            self.playRequested.emit()
        self.is_playing = not self.is_playing
        
    def on_seek(self, value):
        if self.progress_slider.isSliderDown():
            self.seekRequested.emit(value / 1000.0)
            
    def update_time(self, current, total):
        """Update time display"""
        current_str = f"{int(current//60):02d}:{int(current%60):02d}"
        total_str = f"{int(total//60):02d}:{int(total%60):02d}"
        self.time_label.setText(f"{current_str} / {total_str}")
        
        if total > 0:
            self.progress_slider.blockSignals(True)
            self.progress_slider.setMaximum(int(total * 1000))
            self.progress_slider.setValue(int(current * 1000))
            self.progress_slider.blockSignals(False)


class ModernMainWindow(QMainWindow):
    """
    Main window with modern glassmorphic design
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectrogram Analyzer - Modern UI")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize components
        self.audio_data = None
        self.sample_rate = None
        self.spectrogram_data = None
        self.current_file = None
        
        # Thread pool for background tasks
        self.thread_pool = QThreadPool()
        
        self.setup_ui()
        self.apply_theme()
        self.setup_shortcuts()
        
    def setup_ui(self):
        """Setup the modern UI layout"""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top toolbar
        self.setup_toolbar()
        
        # Content area with splitter
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setHandleWidth(1)
        
        # Left panel (file list)
        left_panel = GlassPanel()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)
        
        # File list header
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
        
        add_btn = ModernButton("+")
        add_btn.setFixedSize(32, 32)
        add_btn.clicked.connect(self.add_files)
        header_layout.addWidget(add_btn)
        
        left_layout.addLayout(header_layout)
        
        # File list
        self.file_list = ModernFileList()
        self.file_list.itemDoubleClicked.connect(self.load_selected_file)
        self.file_list.fileDeleteRequested.connect(self.remove_selected_files)
        left_layout.addWidget(self.file_list)
        
        # Add file controls
        file_controls = QHBoxLayout()
        sort_btn = ModernButton("Sort")
        sort_btn.clicked.connect(self.show_sort_menu)
        clear_btn = ModernButton("Clear")
        clear_btn.clicked.connect(self.clear_files)
        file_controls.addWidget(sort_btn)
        file_controls.addWidget(clear_btn)
        left_layout.addLayout(file_controls)
        
        content_splitter.addWidget(left_panel)
        
        # Right panel (spectrogram and controls)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        
        # Spectrogram display area (placeholder for now)
        self.spectrogram_area = GlassPanel()
        spectrogram_layout = QVBoxLayout(self.spectrogram_area)
        
        # Placeholder for spectrogram canvas
        self.spectrogram_placeholder = QLabel("Drop audio files to begin")
        self.spectrogram_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spectrogram_placeholder.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.4);
                font-size: 18px;
                padding: 40px;
            }
        """)
        spectrogram_layout.addWidget(self.spectrogram_placeholder)
        
        right_layout.addWidget(self.spectrogram_area, 1)
        
        # Audio controls
        self.control_bar = ModernControlBar()
        self.control_bar.playRequested.connect(self.play_audio)
        self.control_bar.pauseRequested.connect(self.pause_audio)
        self.control_bar.stopRequested.connect(self.stop_audio)
        right_layout.addWidget(self.control_bar)
        
        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([300, 1100])
        
        main_layout.addWidget(content_splitter)
        
        # Status bar
        self.setup_status_bar()
        
    def setup_toolbar(self):
        """Setup modern toolbar"""
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
        """)
        
        # File operations
        open_action = QAction(qta.icon('fa5s.folder-open', color='#9CA3AF'), "Open", self)
        open_action.triggered.connect(self.add_files)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        # View operations
        zoom_in_action = QAction(qta.icon('fa5s.search-plus', color='#9CA3AF'), "Zoom In", self)
        zoom_out_action = QAction(qta.icon('fa5s.search-minus', color='#9CA3AF'), "Zoom Out", self)
        reset_action = QAction(qta.icon('fa5s.compress', color='#9CA3AF'), "Reset View", self)
        
        toolbar.addAction(zoom_in_action)
        toolbar.addAction(zoom_out_action)
        toolbar.addAction(reset_action)
        
        toolbar.addSeparator()
        
        # Processing operations
        filter_action = QAction(qta.icon('fa5s.filter', color='#9CA3AF'), "Filters", self)
        settings_action = QAction(qta.icon('fa5s.cog', color='#9CA3AF'), "Settings", self)
        
        toolbar.addAction(filter_action)
        toolbar.addAction(settings_action)
        
    def setup_status_bar(self):
        """Setup modern status bar"""
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
        
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # File operations
        QShortcut(QKeySequence("Ctrl+O"), self, self.add_files)
        QShortcut(QKeySequence("Delete"), self.file_list, self.remove_selected_files)
        
        # Playback
        QShortcut(QKeySequence("Space"), self, self.control_bar.toggle_play)
        QShortcut(QKeySequence("S"), self, self.stop_audio)
        
        # View
        QShortcut(QKeySequence("Ctrl++"), self, self.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self, self.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, self.reset_zoom)
        
    def apply_theme(self):
        """Apply the dark glassmorphic theme"""
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
            }
            QScrollBar:vertical {
                background: rgba(255, 255, 255, 0.02);
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 0.15);
            }
            QMenu {
                background: rgba(20, 20, 30, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 4px;
            }
            QMenu::item {
                color: rgba(255, 255, 255, 0.9);
                padding: 8px 16px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background: rgba(99, 102, 241, 0.3);
            }
        """)
        
    def add_files(self):
        """Add audio files to the list"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio Files",
            "",
            "Audio Files (*.wav *.flac);;All Files (*.*)"
        )
        
        for file_path in files:
            if not any(self.file_list.item(i).data(Qt.ItemDataRole.UserRole) == file_path
                      for i in range(self.file_list.count())):
                item = QListWidgetItem(os.path.basename(file_path))
                item.setIcon(qta.icon('fa5s.music', color='#8B5CF6'))
                item.setData(Qt.ItemDataRole.UserRole, file_path)
                self.file_list.addItem(item)
                
    def remove_selected_files(self):
        """Remove selected files from the list"""
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
            
    def clear_files(self):
        """Clear all files from the list"""
        self.file_list.clear()
        
    def show_sort_menu(self):
        """Show sorting options menu"""
        menu = QMenu(self)
        menu.setStyleSheet(self.styleSheet())
        
        # Sort options
        name_action = menu.addAction("Sort by Name")
        time_action = menu.addAction("Sort by Time")
        size_action = menu.addAction("Sort by Size")
        
        menu.addSeparator()
        
        asc_action = menu.addAction("Ascending")
        desc_action = menu.addAction("Descending")
        
        # Execute menu
        menu.exec(self.cursor().pos())
        
    def load_selected_file(self, item):
        """Load the selected audio file"""
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if file_path:
            self.current_file = file_path
            self.statusBar().showMessage(f"Loading: {os.path.basename(file_path)}")
            # Load audio and compute spectrogram (to be implemented)
            
    def play_audio(self):
        """Play the current audio"""
        if self.current_file:
            self.statusBar().showMessage("Playing audio...")
            
    def pause_audio(self):
        """Pause audio playback"""
        self.statusBar().showMessage("Paused")
        
    def stop_audio(self):
        """Stop audio playback"""
        self.control_bar.is_playing = False
        self.control_bar.play_btn.setIcon(qta.icon('fa5s.play', color='white'))
        self.statusBar().showMessage("Stopped")
        
    def zoom_in(self):
        """Zoom in on spectrogram"""
        pass
        
    def zoom_out(self):
        """Zoom out on spectrogram"""
        pass
        
    def reset_zoom(self):
        """Reset spectrogram zoom"""
        pass