# Modern Spectrogram Analyzer

A high-performance audio analysis tool with a modern glassmorphic UI design, built with PySide6 for superior performance and user experience.

## 🎨 Features

### Modern UI Design
- **Glassmorphic Design**: Beautiful glass-effect panels with blur and translucency
- **Gradient Accents**: Cool-to-warm spectrum gradients for visual appeal
- **Smooth Animations**: Material-like transitions and hover effects
- **Dark Theme**: Deep, desaturated background with vibrant accents

### Performance Optimizations
- **JIT Compilation**: Numba-accelerated spectrogram computation
- **Smart Caching**: LRU cache for repeated operations
- **Level-of-Detail Rendering**: Automatic quality adjustment based on zoom
- **Multithreaded Audio**: Dedicated playback thread prevents UI blocking
- **Optimized Zoom**: Smooth zooming constrained to audio boundaries

### Audio Processing
- **Format Support**: FLAC and WAV files
- **Real-time Playback**: Smooth audio playback with position tracking
- **Advanced Filters**:
  - Gain adjustment
  - High-pass filter
  - Low-pass filter
  - Band-pass filter
  - Notch filter
  - Audio normalization

### Spectrogram Features
- **Customizable Parameters**: Window size, overlap, colormap
- **Multiple Colormaps**: Viridis, Plasma, Inferno, Magma, and more
- **Interactive Navigation**: 
  - Smooth pan and zoom
  - Crosshair display (Alt+hover)
  - Click to seek
  - Selection regions

## 🚀 Installation

### Requirements
- Python 3.8 or higher
- Linux, macOS, or Windows

### Quick Install
```bash
# Clone the repository
cd spectrogram_gui

# Run the installer
chmod +x install.sh
./install.sh
```

### Manual Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_new.txt

# Run the application
python main_final.py
```

## 🎮 Usage

### Keyboard Shortcuts
| Shortcut | Action |
|----------|--------|
| `Space` | Play/Pause audio |
| `S` | Stop playback |
| `Ctrl+O` | Open audio files |
| `Ctrl+S` | Export spectrogram |
| `Ctrl++` | Zoom in |
| `Ctrl+-` | Zoom out |
| `Ctrl+0` | Reset view |
| `Delete` | Remove selected files |
| `Alt+hover` | Show crosshair |
| `Ctrl+click` | Seek to position |
| `Ctrl+scroll` | Zoom Y-axis only |
| `Shift+scroll` | Zoom X-axis only |

### Loading Audio Files
1. Click the **+** button or use `Ctrl+O`
2. Select FLAC or WAV files
3. Double-click a file in the list to load it
4. Or drag and drop files directly into the file list

### Adjusting Spectrogram
1. Use the **Controls** panel on the right
2. Adjust window size (512-8192 samples)
3. Set overlap percentage (0-99%)
4. Choose a colormap
5. Click **Update** to apply changes

### Applying Filters
1. Click the **Filter** button in the toolbar
2. Enable desired filters
3. Adjust filter parameters
4. Click **Apply** to process audio

## 🎨 UI Components

### Glass Panels
- Semi-transparent backgrounds with blur effects
- Subtle borders with inner highlights
- 8-16dp border radius for smooth corners
- Micro-shadows for depth perception

### Modern Buttons
- Pill-shaped design with gradients
- Primary buttons: Vibrant gradient (indigo to purple)
- Secondary buttons: Subtle glass effect
- Hover animations with smooth transitions

### Audio Controls
- Glass-effect control bar
- Gradient progress slider
- Volume control with icon
- Time display in monospace font

## 🔧 Technical Details

### Performance Optimizations

#### Spectrogram Computation
- **Numba JIT**: Compiled spectrogram functions for 3-5x speedup
- **Parallel Processing**: Multi-core utilization for large files
- **Smart Caching**: Results cached for repeated computations

#### Rendering
- **Level-of-Detail**: Automatic quality adjustment
- **Viewport Culling**: Only render visible data
- **Smooth Zoom**: Constrained to data boundaries
- **60 FPS Updates**: Smooth animation timer

#### Memory Management
- **Lazy Loading**: Load data only when needed
- **Cache Limits**: Automatic cache size management
- **Efficient Data Types**: Optimized numpy arrays

### Architecture

```
spectrogram_gui/
├── gui_modern/
│   ├── main_window_complete.py  # Main application window
│   ├── spectrogram_canvas.py    # Optimized spectrogram display
│   ├── audio_processor.py       # Audio processing engine
│   └── audio_player.py          # Playback controller
├── main_final.py                # Application entry point
└── requirements_new.txt         # PySide6 dependencies
```

## 🎯 Design Philosophy

### Visual Language
- **Depth without clutter**: Glass effects create layers
- **Calm and modern**: Dark base with vibrant accents
- **Consistent spacing**: 8px grid system
- **Readable typography**: Clear geometric sans-serif

### Interaction Patterns
- **Primary actions obvious**: Prominent buttons and controls
- **Secondary actions tucked**: Icon menus and context menus
- **Immediate feedback**: Toasts and subtle animations
- **Keyboard friendly**: Comprehensive shortcuts

### Accessibility
- **High contrast**: Strong contrast on interactive elements
- **Color + shape**: Visual cues beyond color alone
- **Keyboard navigation**: Full keyboard support
- **Scalable interface**: Responsive to different screen sizes

## 🐛 Troubleshooting

### Audio doesn't play
- Check audio device settings
- Ensure sounddevice is properly installed
- Try different audio output device

### Slow performance
- Reduce window size in spectrogram settings
- Disable level-of-detail if issues persist
- Close other resource-intensive applications

### UI scaling issues
- Application supports high DPI displays
- Adjust system DPI settings if needed
- Use Ctrl+scroll to zoom interface

## 📝 License

This project is provided as-is for audio analysis purposes.

## 🙏 Acknowledgments

Built with:
- **PySide6**: Modern Qt bindings for Python
- **PyQtGraph**: Fast plotting and graphics
- **NumPy/SciPy**: Scientific computing
- **Librosa**: Audio analysis
- **Numba**: JIT compilation

---

**Enjoy your enhanced audio analysis experience!**