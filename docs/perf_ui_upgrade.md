# Performance & UI Upgrade Documentation

## Overview

This document describes the performance optimizations and UI/UX upgrades implemented for the Spectrogram GUI application, migrating from PyQt5 to PySide6 (Qt6) while maintaining identical functionality and improving performance for CPU-only systems.

## ✅ Completed Upgrades

### Phase 1: Bug Fixes & Stability

#### Logging System
- Added lightweight logging utility (`utils/logger.py`) with DEBUG/INFO levels
- Performance timing context manager for profiling operations
- Logs file loading, buffer creation, plot updates, and CSV operations
- Optional debug file output with `--debug` flag or `DEBUG` environment variable

#### Unit Tests
- **test_csv_roundtrip.py**: Validates CSV I/O preserves timestamps correctly
- **test_lod_envelope.py**: Tests Level-of-Detail decimation preserves peaks
- All existing time parsing tests continue to pass

### Phase 2: Performance Optimizations

#### Rendering Optimizations
- **LOD Decimator** (`utils/lod_decimator.py`):
  - Min/max envelope decimation for time series data
  - Preserves peaks and visual features while reducing point count
  - Adaptive decimation based on view range and display width
  - Cache system for frequently used decimation levels

- **PyQtGraph Optimizations**:
  - `clipToView=True` - Only render visible portions
  - `downsample='auto'` with `downsampleMethod='peak'` - Preserve peaks
  - Configurable antialiasing (disabled by default for performance)
  - `autoLevels=False` on ImageItems to avoid redundant calculations
  - Percentile-based level computation for robust display

#### Concurrency
- **QThreadPool Workers** (`utils/worker_thread.py`):
  - Non-blocking audio file loading
  - Background spectrogram computation
  - Progress reporting with cancellation support
  - Specialized workers for audio and spectrogram tasks
  - ThreadPoolManager with configurable thread count

### Phase 3: Qt6/PySide6 Migration & Modern UI

#### PySide6 Migration
- Complete migration from PyQt5 to PySide6
- Updated all imports and Qt6-specific API changes
- Replaced `pyqtSignal` with `Signal`
- Changed `.exec_()` to `.exec()`
- Removed qdarkstyle dependency

#### Modern Theme System
- **Dark Mode by Default**:
  - Custom QPalette with modern color scheme
  - Gradient backgrounds for depth
  - Indigo (#4F46E5) primary, Cyan (#06B6D4) secondary
  - Dark blue-gray backgrounds (#0B0F19 to #111827)

- **Theme Files**:
  - `styles/theme_tokens.json` - Design tokens for consistency
  - `styles/app.qss` - Modern Qt6 stylesheet with gradients
  - Rounded corners (6-8px), subtle shadows, focus rings
  - High-contrast states for accessibility

#### UX Enhancements

##### Context Menus
- Right-click on file list for sorting options
- Sort by: Name, Start Time, End Time, Duration
- Ascending/Descending order
- Settings persist via QSettings

##### Keyboard Shortcuts
| Shortcut | Action |
|----------|--------|
| **File Operations** ||
| Ctrl+O | Open files |
| Ctrl+Shift+O | Open folder |
| Delete | Remove selected file |
| **Navigation** ||
| ← / Alt+← | Previous file |
| → / Alt+→ | Next file |
| **Playback** ||
| Space | Play/Pause |
| **Zoom** ||
| Ctrl+= | Zoom in |
| Ctrl+- | Zoom out |
| Ctrl+0 | Reset zoom |
| **Tools** ||
| Ctrl+S | Spectrogram settings |
| Ctrl+Enter | Run detection |
| Ctrl+M | Toggle mark event |
| Ctrl+Z | Undo |
| **Other** ||
| Ctrl+, | Preferences |
| F1 | Show shortcuts help |

##### Preferences Dialog
- **Performance Tab**:
  - OpenGL acceleration toggle (off by default for Intel iGPU)
  - Antialiasing toggle
  - Decimation threshold and target points
  - Worker thread count
  - Progress dialog visibility

- **Theme Tab**:
  - Theme selection (Dark/Light/System)
  - Spectrogram colormap selection
  - Font size adjustment
  - Window opacity
  - Visual effects toggles

- **Behavior Tab**:
  - Default file sorting
  - Auto-load first file
  - Session persistence
  - Recent files management

## 🔧 Configuration

### Performance Settings

```python
# In preferences or programmatically:
canvas.set_performance_mode(
    use_opengl=False,  # Default for Intel iGPU
    enable_antialiasing=False  # Better performance
)

# LOD Decimator settings
decimator = LODDecimator(
    threshold=50000,  # Points above which to decimate
    max_points=10000  # Target after decimation
)
```

### Theme Customization

The theme can be customized by editing:
- `styles/app.qss` - Main stylesheet
- `styles/theme_tokens.json` - Color and spacing tokens

To switch themes programmatically:
```python
settings = QSettings("SpectrogramGUI", "Preferences")
settings.setValue("theme/style", "Dark (Default)")
```

## 📊 Performance Metrics

### Before Optimization
- Large FLAC files (>100MB): 5-10s load time, UI freezes
- Panning/zooming: Laggy with >100k points
- Memory usage: High with multiple large files

### After Optimization
- Large FLAC files: <2s to first interactive frame (target)
- Panning/zooming: Smooth with LOD decimation
- Memory usage: Reduced via streaming and caching
- Non-blocking UI during file operations

## 🧪 Testing

Run tests to verify optimizations:
```bash
python3 tests/test_time_parse.py
python3 tests/test_csv_roundtrip.py
python3 tests/test_lod_envelope.py
```

## 🚀 Migration Guide

### From PyQt5 to PySide6

1. Update requirements:
```bash
pip uninstall PyQt5 qdarkstyle
pip install PySide6>=6.5.0
```

2. The migration script has already updated all imports

3. Key changes:
   - `pyqtSignal` → `Signal`
   - `.exec_()` → `.exec()`
   - QAction moved from QtWidgets to QtGui

### Compatibility Notes

- **Python 3.8+** required
- **No GPU/CUDA** dependencies
- **Intel iGPU compatible** - OpenGL disabled by default
- **No breaking changes** to:
  - Public APIs
  - CLI flags
  - CSV schemas
  - Save formats

## 🔍 Troubleshooting

### OpenGL Issues
If OpenGL causes problems:
1. Open Preferences (Ctrl+,)
2. Performance tab → Uncheck "Try OpenGL acceleration"
3. Restart application

### Performance Issues
1. Adjust decimation threshold in Preferences
2. Disable antialiasing
3. Reduce worker thread count on low-end systems

### Theme Issues
If UI elements are not visible:
1. Delete `~/.config/SpectrogramGUI/` to reset settings
2. Check `styles/app.qss` for conflicts with legacy styles

## 📝 Future Enhancements

Potential future optimizations:
- WebGL rendering backend for better cross-platform GPU support
- Lazy loading of file metadata
- Compressed spectrogram cache
- Multi-resolution pyramid for spectrograms
- SIMD optimizations for FFT operations

## Guarantee

✅ **No logic/output changes** - All DSP, analysis, and pipeline functionality remains identical. Only performance and visuals have been improved.