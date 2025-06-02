import os
import numpy as np
from datetime import timedelta
import pandas as pd

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QFileDialog,
    QSplitter, QPushButton, QProgressDialog, QHBoxLayout, QSizePolicy
)
from PyQt5.QtCore import Qt

# Import for Matplotlib toolbar
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from doppler_detector.core import DopplerDetector
from doppler_detector.voting import VotingDetector
from doppler_detector.utils import parse_flac_time_from_filename

from gui.spectrogram_canvas import SpectrogramCanvas
from gui.controls_panel import ControlsPanel
from gui.event_editor import EventEditor
from gui.parameters_dialog import ParametersDialog
from gui.autodetect_dialog import AutoDetectDialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto Doppler Detector GUI")

        # 1) Base DopplerDetector
        self.detector = DopplerDetector()

        # 2) Spectrogram Canvas
        self.canvas = SpectrogramCanvas(self, width=8, height=6, dpi=100)

        # 3) Navigation toolbar for Zoom/Pan
        self.nav_toolbar = NavigationToolbar(self.canvas, self)

        # 4) Edit Mode button (Toggle)
        self.edit_mode_button = QPushButton("Edit Mode (Delete Tracks)")
        self.edit_mode_button.setCheckable(True)
        self.edit_mode_button.setToolTip("Toggle event deletion mode")

        # 5) Add Mode button (Toggle)
        self.add_mode_button = QPushButton("Add Mode (Manual Track)")
        self.add_mode_button.setCheckable(True)
        self.add_mode_button.setToolTip("Toggle manual event-adding mode")

        # 6) ControlsPanel
        self.controls = ControlsPanel(self)

        # 7) EventEditor will be instantiated after detection
        self.event_editor = None

        # 8) Splitter ראשי: שמאל = Toolbar + Edit/Add + Canvas; ימין = ControlsPanel
        splitter_main = QSplitter(Qt.Horizontal)
        frame_left = QWidget()
        left_layout = QVBoxLayout()

        # a) Toolbar
        left_layout.addWidget(self.nav_toolbar)

        # b) Edit/Add buttons
        editadd_hbox = QHBoxLayout()
        editadd_hbox.addWidget(self.edit_mode_button)
        editadd_hbox.addWidget(self.add_mode_button)
        left_layout.addLayout(editadd_hbox)

        # c) Spectrogram Canvas
        left_layout.addWidget(self.canvas)

        frame_left.setLayout(left_layout)
        splitter_main.addWidget(frame_left)

        # Controls מימין
        splitter_main.addWidget(self.controls)
        splitter_main.setStretchFactor(0, 4)
        splitter_main.setStretchFactor(1, 1)
        self.setCentralWidget(splitter_main)

        # 9) Connect signals to controls
        self.controls.open_button.clicked.connect(self.open_file)
        self.controls.run_button.clicked.connect(self.run_detection)
        self.controls.save_csv_button.clicked.connect(self.save_results_csv)
        self.controls.params_button.clicked.connect(self.open_parameters_dialog)
        self.controls.auto_button.clicked.connect(self.open_autodetect_dialog)

        # Undo/Add
        self.controls.undo_button.clicked.connect(self.on_undo_clicked)
        self.controls.add_button.clicked.connect(self.on_add_clicked)

        # edit/add toggle
        self.edit_mode_button.clicked.connect(self.on_edit_toggle)
        self.add_mode_button.clicked.connect(self.on_add_toggle)

        # 10) State variables
        self.current_filepath = None
        self.start_time = None   # datetime from FLAC filename
        self.detected_tracks = []
        self.freqs = None
        self.times = None
        self.Sxx_norm = None
        self.Sxx_filt = None

        # 11) Default param grid for Voting
        self.auto_param_grid = [
            {
                "POWER_THRESHOLD": 0.1,
                "PEAK_PROMINENCE": 0.04,
                "MAX_FREQ_JUMP_HZ": 8,
                "MIN_TRACK_LENGTH_FRAMES": 6,
                "MAX_TRACK_FREQ_STD_HZ": 70,
                "GAP_PROMINENCE_FACTOR": 0.5,
                "GAP_POWER_FACTOR": 0.8
            },
            {
                "POWER_THRESHOLD": 0.5,
                "PEAK_PROMINENCE": 0.06,
                "MAX_FREQ_JUMP_HZ": 15,
                "MIN_TRACK_LENGTH_FRAMES": 13,
                "MAX_TRACK_FREQ_STD_HZ": 70,
                "GAP_PROMINENCE_FACTOR": 0.8,
                "GAP_POWER_FACTOR": 0.8
            }
        ]

    def on_edit_toggle(self, checked):
        """
        Toggle Delete Mode in EventEditor.
        """
        if self.event_editor:
            self.event_editor.set_delete_mode(checked)
        if checked:
            # turn off Add mode if on
            if self.add_mode_button.isChecked():
                self.add_mode_button.setChecked(False)
                if self.event_editor:
                    self.event_editor.set_add_mode(False)

    def on_add_toggle(self, checked):
        """
        Toggle Add Mode in EventEditor.
        """
        if self.event_editor:
            self.event_editor.set_add_mode(checked)
        if checked:
            # turn off Delete mode if on
            if self.edit_mode_button.isChecked():
                self.edit_mode_button.setChecked(False)
                if self.event_editor:
                    self.event_editor.set_delete_mode(False)

    def on_undo_clicked(self):
        """
        Undo last edit in EventEditor.
        """
        if self.event_editor:
            self.event_editor.undo()

    def on_add_clicked(self):
        """
        Equivalent to toggling Add Mode button.
        """
        current = self.add_mode_button.isChecked()
        self.add_mode_button.setChecked(not current)
        self.on_add_toggle(not current)

    def open_file(self):
        """
        Open a FLAC/WAV file, parse its filename for start_time, compute & display spectrogram.
        """
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Open FLAC or WAV",
            os.getcwd(),
            "Audio Files (*.flac *.wav)"
        )
        if not fname:
            return

        self.current_filepath = fname
        self.start_time = parse_flac_time_from_filename(fname)

        y, sr = self.detector.load_audio(fname)
        freqs, times, Sxx_norm, Sxx_filt = self.detector.compute_spectrogram(y, sr)

        self.freqs = freqs
        self.times = times
        self.Sxx_norm = Sxx_norm
        self.Sxx_filt = Sxx_filt

        # Plot spectrogram (automatically stores all for hover)
        self.canvas.plot_spectrogram(freqs, times, Sxx_norm, start_time=self.start_time)

    def open_parameters_dialog(self):
        """
        Open the Detection Parameters dialog. If OK is clicked,
        the detector instance is updated accordingly.
        """
        dlg = ParametersDialog(self, detector=self.detector)
        dlg.resize(500, 600)
        dlg.exec_()

    def open_autodetect_dialog(self):
        """
        Open the Auto-Detect (Voting) Settings dialog. If OK is clicked,
        update self.auto_param_grid from the dialog.
        """
        dlg = AutoDetectDialog(self, param_grid=self.auto_param_grid)
        dlg.resize(600, 400)
        if dlg.exec_():
            self.auto_param_grid = dlg.param_grid

    def run_detection(self):
        """
        Run detection. If Use Auto-Voting is checked, run VotingDetector with
        optional multiprocessing settings. Otherwise, run a single-configuration detection.
        """
        if not self.current_filepath:
            return

        use_voting = self.controls.voting_checkbox.isChecked()
        enable_mp = self.controls.mp_checkbox.isChecked()
        num_cores = self.controls.num_cores_spin.value() if (use_voting and enable_mp) else None

        if use_voting:
            voting_detector = VotingDetector(
                self.detector,
                self.auto_param_grid,
                num_processes=num_cores
            )
            # Create a progress dialog with a range of [0, N_RUNS]
            progress = QProgressDialog("Initializing...", None, 0, voting_detector.N_RUNS, self)
            progress.setWindowModality(Qt.ApplicationModal)
            progress.setCancelButton(None)
            progress.setMinimumDuration(0)
            progress.setWindowTitle("Running Auto-Detect")

            # Call run_voting with the progress dialog to receive updates
            freqs, times, Sxx_norm, tracks = voting_detector.run_voting(
                self.current_filepath,
                progress_dialog=progress
            )

            progress.close()
        else:
            # Single-configuration detection
            freqs, times, Sxx_norm, Sxx_filt = self.detector.compute_spectrogram(
                *self.detector.load_audio(self.current_filepath)
            )
            tracks = self.detector.run_detection(self.current_filepath)

        self.freqs = freqs
        self.times = times
        self.Sxx_norm = Sxx_norm
        self.detected_tracks = tracks

        # Re-plot spectrogram and tracks
        self.canvas.plot_spectrogram(freqs, times, Sxx_norm, start_time=self.start_time)
        self.canvas.plot_tracks(freqs, times, tracks, start_time=self.start_time)

        # Initialize EventEditor and set tracks
        self.event_editor = EventEditor(self.canvas, freqs, times, start_time=self.start_time)
        self.event_editor.set_tracks(tracks)

    def save_results_csv(self):
        """
        Export a CSV with one row per detected event. Columns:
          - EventID (1-based index)
          - StartTime[s] (relative)
          - EndTime[s]   (relative)
          - StartTimestamp (absolute)
          - EndTimestamp   (absolute)
          - NumHarmonics
          - FileStartTimestamp (absolute, same for all rows)
        """
        if not self.detected_tracks or not self.current_filepath:
            return

        harmonic_count = self.controls.harmonics_spin.value()
        file_start_dt = self.start_time

        records = []
        dt_frame = self.times[1] - self.times[0]

        for i, track in enumerate(self.detected_tracks, start=1):
            first_ti = track[0][0]
            last_ti = track[-1][0]
            start_time_sec = first_ti * dt_frame
            end_time_sec = last_ti * dt_frame

            if file_start_dt is not None:
                abs_start = file_start_dt + timedelta(seconds=start_time_sec)
                abs_end = file_start_dt + timedelta(seconds=end_time_sec)
                start_ts_str = abs_start.isoformat(sep=" ")
                end_ts_str = abs_end.isoformat(sep=" ")
                file_timestamp_str = file_start_dt.isoformat(sep=" ")
            else:
                start_ts_str = ""
                end_ts_str = ""
                file_timestamp_str = ""

            records.append({
                "EventID": i,
                "StartTime[s]": start_time_sec,
                "EndTime[s]": end_time_sec,
                "StartTimestamp": start_ts_str,
                "EndTimestamp": end_ts_str,
                "NumHarmonics": harmonic_count,
                "FileStartTimestamp": file_timestamp_str
            })

        df = pd.DataFrame.from_records(records)
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV",
            os.getcwd(),
            "CSV files (*.csv)"
        )
        if save_path:
            df.to_csv(save_path, index=False)
