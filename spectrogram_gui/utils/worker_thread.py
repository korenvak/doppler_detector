"""
QThreadPool workers for non-blocking operations.
Handles file loading, spectrogram computation, and other heavy tasks.
"""

from PySide6.QtCore import QRu, Signalnnable, QObject, Signal, QThreadPool
import traceback
import sys
from typing import Optional, Callable, Any, Tuple
from doppler_detector.spectrogram_gui.utils.logger import debug, info, error, timer


class WorkerSignals(QObject):
    """
    Signals for worker thread communication.
    """
    started = Signal()
    finished = Signal()
    error = Signal(tuple)  # (exctype, value, traceback)
    result = Signal(object)
    progress = Signal(int)  # Progress percentage 0-100


class Worker(QRunnable):
    """
    Generic worker thread for running functions in the background.
    """
    
    def __init__(self, fn: Callable, *args, **kwargs):
        """
        Initialize worker with function and arguments.
        
        Args:
            fn: Function to run in background
            *args: Positional arguments for fn
            **kwargs: Keyword arguments for fn
        """
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self._is_cancelled = False
        
    def cancel(self):
        """Cancel the worker operation."""
        self._is_cancelled = True
        debug("Worker cancelled")
        
    @property
    def is_cancelled(self):
        """Check if worker has been cancelled."""
        return self._is_cancelled
        
    def run(self):
        """Execute the function with error handling."""
        try:
            self.signals.started.emit()
            debug(f"Worker started: {self.fn.__name__}")
            
            # Add progress callback to kwargs if function supports it
            if 'progress_callback' in self.kwargs:
                self.kwargs['progress_callback'] = self.signals.progress.emit
                
            # Check for cancellation in kwargs
            if 'is_cancelled' in self.kwargs:
                self.kwargs['is_cancelled'] = lambda: self._is_cancelled
                
            with timer(f"Worker task: {self.fn.__name__}"):
                result = self.fn(*self.args, **self.kwargs)
                
            if not self._is_cancelled:
                self.signals.result.emit(result)
                debug(f"Worker completed: {self.fn.__name__}")
        except Exception as e:
            error(f"Worker error in {self.fn.__name__}: {e}")
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.finished.emit()


class AudioLoadWorker(QRunnable):
    """
    Specialized worker for loading audio files.
    """
    
    def __init__(self, filepath: str, hp: Optional[float] = None, 
                 lp: Optional[float] = None, gain_db: float = 0):
        super().__init__()
        self.filepath = filepath
        self.hp = hp
        self.lp = lp
        self.gain_db = gain_db
        self.signals = WorkerSignals()
        self._is_cancelled = False
        
    def cancel(self):
        """Cancel the loading operation."""
        self._is_cancelled = True
        
    def run(self):
        """Load audio file in background."""
        try:
            self.signals.started.emit()
            info(f"Loading audio: {self.filepath}")
            
            from doppler_detector.spectrogram_gui.utils.audio_utils import load_audio_with_filters
            
            # Simulate progress for large files
            self.signals.progress.emit(10)
            
            if self._is_cancelled:
                return
                
            # Load audio
            y, sr = load_audio_with_filters(
                self.filepath, 
                hp=self.hp, 
                lp=self.lp, 
                gain_db=self.gain_db
            )
            
            if self._is_cancelled:
                return
                
            self.signals.progress.emit(100)
            self.signals.result.emit((y, sr, self.filepath))
            
        except Exception as e:
            error(f"Failed to load audio: {e}")
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.finished.emit()


class SpectrogramWorker(QRunnable):
    """
    Specialized worker for computing spectrograms.
    """
    
    def __init__(self, audio_data: Any, sample_rate: int, 
                 filepath: str, params: dict):
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.filepath = filepath
        self.params = params
        self.signals = WorkerSignals()
        self._is_cancelled = False
        
    def cancel(self):
        """Cancel the computation."""
        self._is_cancelled = False
        
    def run(self):
        """Compute spectrogram in background."""
        try:
            self.signals.started.emit()
            info(f"Computing spectrogram for {self.filepath}")
            
            from doppler_detector.spectrogram_gui.utils.spectrogram_utils import compute_spectrogram
            
            # Progress updates
            self.signals.progress.emit(20)
            
            if self._is_cancelled:
                return
                
            # Compute spectrogram
            freqs, times, Sxx, params_used = compute_spectrogram(
                self.audio_data,
                self.sample_rate,
                self.filepath,
                params=self.params
            )
            
            if self._is_cancelled:
                return
                
            self.signals.progress.emit(100)
            self.signals.result.emit((freqs, times, Sxx, params_used))
            
        except Exception as e:
            error(f"Failed to compute spectrogram: {e}")
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        finally:
            self.signals.finished.emit()


class ThreadPoolManager:
    """
    Manager for QThreadPool with convenient methods.
    """
    
    def __init__(self, max_threads: Optional[int] = None):
        """
        Initialize thread pool manager.
        
        Args:
            max_threads: Maximum number of threads (None for auto)
        """
        self.pool = QThreadPool.globalInstance()
        if max_threads:
            self.pool.setMaxThreadCount(max_threads)
        else:
            # Use half of available cores for background tasks
            import multiprocessing
            cores = multiprocessing.cpu_count()
            self.pool.setMaxThreadCount(max(1, cores // 2))
            
        self.active_workers = []
        debug(f"ThreadPool initialized with {self.pool.maxThreadCount()} threads")
        
    def submit(self, worker: QRunnable) -> QRunnable:
        """
        Submit a worker to the thread pool.
        
        Args:
            worker: Worker to execute
            
        Returns:
            The submitted worker
        """
        self.active_workers.append(worker)
        
        # Clean up finished workers
        self.active_workers = [w for w in self.active_workers 
                              if hasattr(w, '_is_cancelled') and not w._is_cancelled]
        
        self.pool.start(worker)
        debug(f"Worker submitted, {self.pool.activeThreadCount()} active threads")
        return worker
        
    def cancel_all(self):
        """Cancel all active workers."""
        for worker in self.active_workers:
            if hasattr(worker, 'cancel'):
                worker.cancel()
        self.active_workers.clear()
        debug("All workers cancelled")
        
    def wait_for_done(self, msecs: int = -1) -> bool:
        """
        Wait for all threads to complete.
        
        Args:
            msecs: Timeout in milliseconds (-1 for infinite)
            
        Returns:
            True if all threads completed, False if timeout
        """
        return self.pool.waitForDone(msecs)