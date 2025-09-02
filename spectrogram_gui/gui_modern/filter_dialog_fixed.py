"""
Fixed Modern Filter Dialog with proper integration
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QMessageBox, QCheckBox,
    QDoubleSpinBox, QComboBox
)
from PySide6.QtCore import Qt
import numpy as np
from scipy.signal import butter, sosfilt


class ModernFilterDialog(QDialog):
    """
    Modern filter dialog that properly integrates with the main window
    """
    
    def __init__(self, main_window, mode="bandpass"):
        super().__init__(main_window)
        self.main = main_window
        self.mode = mode
        self.setWindowTitle(f"{mode.capitalize()} Filter")
        self.setModal(True)
        self.resize(400, 200)
        
        # Apply modern styling
        self.setStyleSheet("""
            QDialog {
                background: rgba(20, 20, 30, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
            }
            QLabel {
                color: rgba(255, 255, 255, 0.8);
            }
            QLineEdit, QDoubleSpinBox {
                background: rgba(255, 255, 255, 0.05);
                color: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                padding: 4px 8px;
                min-width: 100px;
            }
            QLineEdit:hover, QDoubleSpinBox:hover {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(139, 92, 246, 0.3);
            }
            QCheckBox {
                color: rgba(255, 255, 255, 0.8);
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
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
            QPushButton {
                background: rgba(255, 255, 255, 0.05);
                color: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(139, 92, 246, 0.5);
            }
            QPushButton#primaryButton {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #6366F1,
                    stop: 1 #8B5CF6
                );
                color: white;
                border: none;
            }
            QPushButton#primaryButton:hover {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #7C7FFF,
                    stop: 1 #9F6FFF
                );
            }
        """)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout(self)
        
        # Frequency cutoff inputs
        freq_layout = QHBoxLayout()
        
        if self.mode != "lowpass":
            freq_layout.addWidget(QLabel("Low cutoff (Hz):"))
            self.low_edit = QLineEdit("100")
            freq_layout.addWidget(self.low_edit)
            
        if self.mode != "highpass":
            freq_layout.addWidget(QLabel("High cutoff (Hz):"))
            self.high_edit = QLineEdit("3000")
            freq_layout.addWidget(self.high_edit)
            
        layout.addLayout(freq_layout)
        
        # TV Denoising option
        tv_row = QHBoxLayout()
        self.tv_chk = QCheckBox("TV Denoising")
        tv_row.addWidget(self.tv_chk)
        tv_row.addWidget(QLabel("Weight:"))
        
        self.tv_weight_spin = QDoubleSpinBox()
        self.tv_weight_spin.setRange(0.05, 0.3)
        self.tv_weight_spin.setSingleStep(0.01)
        self.tv_weight_spin.setValue(0.1)
        tv_row.addWidget(self.tv_weight_spin)
        
        layout.addLayout(tv_row)
        
        # Buttons
        btns = QHBoxLayout()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(cancel_btn)
        
        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("primaryButton")
        apply_btn.clicked.connect(self.apply_filter)
        btns.addWidget(apply_btn)
        
        layout.addLayout(btns)
        
    def apply_filter(self):
        """Apply the filter to the audio"""
        try:
            if self.mode != "lowpass":
                low = float(self.low_edit.text())
            if self.mode != "highpass":
                high = float(self.high_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid", "Enter valid cutoff frequencies.")
            return
            
        # Get selected range from canvas
        canvas = getattr(self.main, 'canvas', None) or getattr(self.main, 'spectrogram_canvas', None)
        if not canvas:
            QMessageBox.warning(self, "Error", "No spectrogram canvas found.")
            return
            
        sel = getattr(canvas, 'selected_range', None)
        if sel is None:
            # Apply to entire duration when no range selected
            if hasattr(canvas, 'times') and canvas.times is not None:
                t0, t1 = canvas.times[0], canvas.times[-1]
            else:
                QMessageBox.warning(self, "Error", "No audio data loaded.")
                return
        else:
            t0, t1 = sel
            
        # Get audio player
        audio_player = getattr(self.main, 'audio_player', None)
        if not audio_player:
            QMessageBox.warning(self, "Error", "No audio player found.")
            return
            
        # Get waveform and sample rate
        if hasattr(audio_player, 'get_waveform_copy'):
            wave, sr = audio_player.get_waveform_copy(return_sr=True)
        elif hasattr(audio_player, 'audio_data') and hasattr(audio_player, 'sample_rate'):
            wave = audio_player.audio_data.copy() if audio_player.audio_data is not None else None
            sr = audio_player.sample_rate
        else:
            QMessageBox.warning(self, "Error", "Cannot access audio data.")
            return
            
        if wave is None or sr is None:
            QMessageBox.warning(self, "Error", "No audio loaded.")
            return
            
        total = len(wave) / sr
        i0 = int((t0 / total) * len(wave))
        i1 = int((t1 / total) * len(wave))
        
        if i1 <= i0:
            QMessageBox.warning(self, "Invalid Range", "Selected range is invalid.")
            return
            
        # Design and apply filter
        try:
            if self.mode == "highpass":
                sos = butter(4, low, btype="highpass", fs=sr, output="sos")
            elif self.mode == "lowpass":
                sos = butter(4, high, btype="lowpass", fs=sr, output="sos")
            else:  # bandpass
                sos = butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
        except Exception as e:
            QMessageBox.warning(self, "Filter Error", f"Failed to design filter: {str(e)}")
            return
            
        # Backup for undo (if main window supports it)
        if hasattr(self.main, 'add_undo_action') and hasattr(canvas, 'Sxx_raw'):
            prev_sxx = canvas.Sxx_raw.copy() if hasattr(canvas, 'Sxx_raw') else None
            prev_times = canvas.times.copy() if hasattr(canvas, 'times') else None
            prev_freqs = canvas.freqs.copy() if hasattr(canvas, 'freqs') else None
            prev_start = getattr(canvas, 'start_time', None)
            self.main.add_undo_action(("waveform", (wave.copy(), prev_sxx, prev_times, prev_freqs, prev_start)))
            
        # Apply filter
        seg = wave[i0:i1].copy()
        filtered = sosfilt(sos, seg)
        new_wave = wave.copy()
        new_wave[i0:i1] = filtered
        
        # Replace waveform in audio player
        if hasattr(audio_player, 'replace_waveform'):
            audio_player.replace_waveform(new_wave)
        elif hasattr(audio_player, 'set_audio_data'):
            audio_player.set_audio_data(new_wave, sr)
        else:
            audio_player.audio_data = new_wave
            audio_player.sample_rate = sr
            
        # Recompute and plot spectrogram
        try:
            # Try to use the audio processor if available
            if hasattr(self.main, 'audio_processor'):
                processor = self.main.audio_processor
                if hasattr(processor, 'compute_spectrogram'):
                    freqs, times, Sxx = processor.compute_spectrogram(new_wave, sr)
                    
                    # Apply TV denoising if checked
                    if self.tv_chk.isChecked():
                        try:
                            from spectrogram_gui.utils.filter_utils import apply_tv_denoising_2d
                            Sxx = apply_tv_denoising_2d(Sxx, weight=self.tv_weight_spin.value())
                        except ImportError:
                            pass
                    
                    # Update the canvas
                    if hasattr(canvas, 'set_spectrogram_data'):
                        canvas.set_spectrogram_data(new_wave, sr, freqs, times, Sxx)
                    elif hasattr(canvas, 'plot_spectrogram'):
                        start_time = getattr(canvas, 'start_time', None)
                        canvas.plot_spectrogram(freqs, times, Sxx, start_time, maintain_view=True)
            else:
                # Fallback: try to use compute_spectrogram directly
                from spectrogram_gui.utils.spectrogram_utils import compute_spectrogram
                
                params = getattr(self.main, 'spectrogram_params', {})
                freqs, times, Sxx, _ = compute_spectrogram(new_wave, sr, "", params=params)
                
                # Apply TV denoising if checked
                if self.tv_chk.isChecked():
                    try:
                        from spectrogram_gui.utils.filter_utils import apply_tv_denoising_2d
                        Sxx = apply_tv_denoising_2d(Sxx, weight=self.tv_weight_spin.value())
                    except ImportError:
                        pass
                
                # Update the canvas
                if hasattr(canvas, 'set_spectrogram_data'):
                    canvas.set_spectrogram_data(new_wave, sr, freqs, times, Sxx)
                elif hasattr(canvas, 'plot_spectrogram'):
                    start_time = getattr(canvas, 'start_time', None)
                    canvas.plot_spectrogram(freqs, times, Sxx, start_time, maintain_view=True)
                    
        except Exception as e:
            QMessageBox.warning(self, "Update Error", f"Failed to update spectrogram: {str(e)}")
            
        self.accept()