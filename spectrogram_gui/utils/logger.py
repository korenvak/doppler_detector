"""
Lightweight logging utility for debugging and performance monitoring.
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Create logger
logger = logging.getLogger('SpectrogramGUI')
logger.setLevel(logging.DEBUG)

# Create formatters
debug_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

info_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Console handler (INFO and above)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(info_formatter)

# File handler (DEBUG and above) - optional, only if debug mode is enabled
file_handler = None
if '--debug' in sys.argv or 'DEBUG' in os.environ:
    log_dir = Path.home() / '.spectrogram_gui' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(debug_formatter)
    logger.addHandler(file_handler)

logger.addHandler(console_handler)

# Convenience functions
def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs)

def critical(msg, *args, **kwargs):
    logger.critical(msg, *args, **kwargs)

# Performance timing context manager
import time
from contextlib import contextmanager

@contextmanager
def timer(operation_name):
    """Context manager for timing operations."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if elapsed > 0.1:  # Only log operations taking > 100ms
            info(f"{operation_name} took {elapsed:.3f}s")
        else:
            debug(f"{operation_name} took {elapsed:.3f}s")