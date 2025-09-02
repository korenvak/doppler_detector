"""
Preferences dialog for performance and theme settings.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
    QWidget, QLabel, QCheckBox, QComboBox, QSpinBox,
    QPushButton, QGroupBox, QFormLayout, QSlider,
    QDialogButtonBox, QMessageBox
)
from PySide6.QtCore import Qt, QSettings, Signal
from PySide6.QtGui import QFont


class PreferencesDialog(QDialog):
    """
    Preferences dialog with tabs for different settings categories.
    """
    
    settings_changed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setModal(True)
        self.resize(600, 500)
        
        # Load settings
        self.settings = QSettings("SpectrogramGUI", "Preferences")
        
        # Create UI
        self.init_ui()
        
        # Load current settings
        self.load_settings()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        
        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_performance_tab()
        self.create_theme_tab()
        self.create_behavior_tab()
        
        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply_settings)
        layout.addWidget(buttons)
        
    def create_performance_tab(self):
        """Create the Performance settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Rendering group
        render_group = QGroupBox("Rendering")
        render_layout = QFormLayout()
        
        # OpenGL acceleration
        self.opengl_check = QCheckBox("Try OpenGL acceleration (experimental)")
        self.opengl_check.setToolTip(
            "Enable OpenGL rendering. May not work on all systems.\n"
            "Falls back to CPU rendering if unavailable."
        )
        render_layout.addRow("GPU Acceleration:", self.opengl_check)
        
        # Antialiasing
        self.antialias_check = QCheckBox("Enable antialiasing")
        self.antialias_check.setToolTip(
            "Smooth edges for better visual quality.\n"
            "May impact performance on large datasets."
        )
        render_layout.addRow("Antialiasing:", self.antialias_check)
        
        # Decimation threshold
        self.decimation_spin = QSpinBox()
        self.decimation_spin.setRange(10000, 1000000)
        self.decimation_spin.setSingleStep(10000)
        self.decimation_spin.setSuffix(" points")
        self.decimation_spin.setToolTip(
            "Number of points above which decimation is applied.\n"
            "Lower values improve performance but may reduce detail."
        )
        render_layout.addRow("Decimation threshold:", self.decimation_spin)
        
        # Target points after decimation
        self.target_points_spin = QSpinBox()
        self.target_points_spin.setRange(1000, 50000)
        self.target_points_spin.setSingleStep(1000)
        self.target_points_spin.setSuffix(" points")
        self.target_points_spin.setToolTip(
            "Target number of points to display after decimation.\n"
            "Higher values show more detail but use more resources."
        )
        render_layout.addRow("Target points:", self.target_points_spin)
        
        render_group.setLayout(render_layout)
        layout.addWidget(render_group)
        
        # Threading group
        thread_group = QGroupBox("Background Processing")
        thread_layout = QFormLayout()
        
        # Thread count
        self.thread_count_spin = QSpinBox()
        self.thread_count_spin.setRange(1, 8)
        self.thread_count_spin.setToolTip(
            "Number of background threads for file loading.\n"
            "More threads can speed up loading but use more CPU."
        )
        thread_layout.addRow("Worker threads:", self.thread_count_spin)
        
        # Progress dialogs
        self.show_progress_check = QCheckBox("Show progress dialogs")
        self.show_progress_check.setToolTip(
            "Display progress bars for long operations."
        )
        thread_layout.addRow("Progress indicators:", self.show_progress_check)
        
        thread_group.setLayout(thread_layout)
        layout.addWidget(thread_group)
        
        layout.addStretch()
        self.tabs.addTab(tab, "Performance")
        
    def create_theme_tab(self):
        """Create the Theme settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Appearance group
        appear_group = QGroupBox("Appearance")
        appear_layout = QFormLayout()
        
        # Theme selection
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark (Default)", "Light", "System"])
        self.theme_combo.setToolTip(
            "Choose the application theme.\n"
            "System follows your OS theme settings."
        )
        appear_layout.addRow("Theme:", self.theme_combo)
        
        # Colormap for spectrograms
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "viridis", "plasma", "inferno", "magma",
            "cividis", "turbo", "hot", "cool",
            "gray", "bone", "copper", "jet"
        ])
        self.colormap_combo.setToolTip(
            "Default colormap for spectrogram display."
        )
        appear_layout.addRow("Spectrogram colormap:", self.colormap_combo)
        
        # Font size
        self.font_size_slider = QSlider(Qt.Horizontal)
        self.font_size_slider.setRange(10, 18)
        self.font_size_slider.setTickPosition(QSlider.TicksBelow)
        self.font_size_slider.setTickInterval(1)
        self.font_size_label = QLabel("14px")
        self.font_size_slider.valueChanged.connect(
            lambda v: self.font_size_label.setText(f"{v}px")
        )
        font_layout = QHBoxLayout()
        font_layout.addWidget(self.font_size_slider)
        font_layout.addWidget(self.font_size_label)
        appear_layout.addRow("Font size:", font_layout)
        
        # Window opacity
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(50, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setTickPosition(QSlider.TicksBelow)
        self.opacity_slider.setTickInterval(10)
        self.opacity_label = QLabel("100%")
        self.opacity_slider.valueChanged.connect(
            lambda v: self.opacity_label.setText(f"{v}%")
        )
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(self.opacity_slider)
        opacity_layout.addWidget(self.opacity_label)
        appear_layout.addRow("Window opacity:", opacity_layout)
        
        appear_group.setLayout(appear_layout)
        layout.addWidget(appear_group)
        
        # Effects group
        effects_group = QGroupBox("Visual Effects")
        effects_layout = QFormLayout()
        
        # Animations
        self.animations_check = QCheckBox("Enable animations")
        self.animations_check.setToolTip(
            "Smooth transitions and animations.\n"
            "Disable for better performance on slower systems."
        )
        effects_layout.addRow("Animations:", self.animations_check)
        
        # Shadows
        self.shadows_check = QCheckBox("Enable shadows")
        self.shadows_check.setToolTip(
            "Drop shadows for depth effect.\n"
            "May impact performance on Intel iGPU."
        )
        effects_layout.addRow("Shadows:", self.shadows_check)
        
        effects_group.setLayout(effects_layout)
        layout.addWidget(effects_group)
        
        layout.addStretch()
        self.tabs.addTab(tab, "Theme")
        
    def create_behavior_tab(self):
        """Create the Behavior settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # File handling group
        file_group = QGroupBox("File Handling")
        file_layout = QFormLayout()
        
        # Default sort
        self.sort_combo = QComboBox()
        self.sort_combo.addItems([
            "Name", "Start Time", "End Time", "Duration"
        ])
        self.sort_combo.setToolTip(
            "Default sorting for file list."
        )
        file_layout.addRow("Default sort:", self.sort_combo)
        
        # Sort order
        self.sort_ascending_check = QCheckBox("Ascending order")
        file_layout.addRow("Sort order:", self.sort_ascending_check)
        
        # Auto-load
        self.auto_load_check = QCheckBox("Auto-load first file")
        self.auto_load_check.setToolTip(
            "Automatically load the first file when opening a folder."
        )
        file_layout.addRow("Auto-load:", self.auto_load_check)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Session group
        session_group = QGroupBox("Session")
        session_layout = QFormLayout()
        
        # Remember last session
        self.remember_session_check = QCheckBox("Remember last session")
        self.remember_session_check.setToolTip(
            "Restore window size and position on startup."
        )
        session_layout.addRow("Session:", self.remember_session_check)
        
        # Remember recent files
        self.remember_files_check = QCheckBox("Remember recent files")
        self.remember_files_check.setToolTip(
            "Keep a list of recently opened files."
        )
        session_layout.addRow("Recent files:", self.remember_files_check)
        
        # Max recent files
        self.max_recent_spin = QSpinBox()
        self.max_recent_spin.setRange(5, 50)
        self.max_recent_spin.setValue(10)
        self.max_recent_spin.setToolTip(
            "Maximum number of recent files to remember."
        )
        session_layout.addRow("Max recent:", self.max_recent_spin)
        
        session_group.setLayout(session_layout)
        layout.addWidget(session_group)
        
        layout.addStretch()
        self.tabs.addTab(tab, "Behavior")
        
    def load_settings(self):
        """Load settings from QSettings."""
        # Performance
        self.opengl_check.setChecked(
            self.settings.value("performance/opengl", False, type=bool)
        )
        self.antialias_check.setChecked(
            self.settings.value("performance/antialiasing", False, type=bool)
        )
        self.decimation_spin.setValue(
            self.settings.value("performance/decimation_threshold", 50000, type=int)
        )
        self.target_points_spin.setValue(
            self.settings.value("performance/target_points", 10000, type=int)
        )
        self.thread_count_spin.setValue(
            self.settings.value("performance/thread_count", 2, type=int)
        )
        self.show_progress_check.setChecked(
            self.settings.value("performance/show_progress", True, type=bool)
        )
        
        # Theme
        self.theme_combo.setCurrentText(
            self.settings.value("theme/style", "Dark (Default)")
        )
        self.colormap_combo.setCurrentText(
            self.settings.value("theme/colormap", "magma")
        )
        self.font_size_slider.setValue(
            self.settings.value("theme/font_size", 14, type=int)
        )
        self.opacity_slider.setValue(
            self.settings.value("theme/opacity", 100, type=int)
        )
        self.animations_check.setChecked(
            self.settings.value("theme/animations", True, type=bool)
        )
        self.shadows_check.setChecked(
            self.settings.value("theme/shadows", True, type=bool)
        )
        
        # Behavior
        self.sort_combo.setCurrentText(
            self.settings.value("behavior/default_sort", "Name")
        )
        self.sort_ascending_check.setChecked(
            self.settings.value("behavior/sort_ascending", True, type=bool)
        )
        self.auto_load_check.setChecked(
            self.settings.value("behavior/auto_load", False, type=bool)
        )
        self.remember_session_check.setChecked(
            self.settings.value("behavior/remember_session", True, type=bool)
        )
        self.remember_files_check.setChecked(
            self.settings.value("behavior/remember_files", True, type=bool)
        )
        self.max_recent_spin.setValue(
            self.settings.value("behavior/max_recent", 10, type=int)
        )
        
    def save_settings(self):
        """Save settings to QSettings."""
        # Performance
        self.settings.setValue("performance/opengl", self.opengl_check.isChecked())
        self.settings.setValue("performance/antialiasing", self.antialias_check.isChecked())
        self.settings.setValue("performance/decimation_threshold", self.decimation_spin.value())
        self.settings.setValue("performance/target_points", self.target_points_spin.value())
        self.settings.setValue("performance/thread_count", self.thread_count_spin.value())
        self.settings.setValue("performance/show_progress", self.show_progress_check.isChecked())
        
        # Theme
        self.settings.setValue("theme/style", self.theme_combo.currentText())
        self.settings.setValue("theme/colormap", self.colormap_combo.currentText())
        self.settings.setValue("theme/font_size", self.font_size_slider.value())
        self.settings.setValue("theme/opacity", self.opacity_slider.value())
        self.settings.setValue("theme/animations", self.animations_check.isChecked())
        self.settings.setValue("theme/shadows", self.shadows_check.isChecked())
        
        # Behavior
        self.settings.setValue("behavior/default_sort", self.sort_combo.currentText())
        self.settings.setValue("behavior/sort_ascending", self.sort_ascending_check.isChecked())
        self.settings.setValue("behavior/auto_load", self.auto_load_check.isChecked())
        self.settings.setValue("behavior/remember_session", self.remember_session_check.isChecked())
        self.settings.setValue("behavior/remember_files", self.remember_files_check.isChecked())
        self.settings.setValue("behavior/max_recent", self.max_recent_spin.value())
        
        self.settings.sync()
        
    def apply_settings(self):
        """Apply settings without closing dialog."""
        self.save_settings()
        self.settings_changed.emit()
        QMessageBox.information(self, "Settings Applied", 
                              "Settings have been applied.\nSome changes may require restart.")
        
    def accept(self):
        """Save settings and close dialog."""
        self.save_settings()
        self.settings_changed.emit()
        super().accept()