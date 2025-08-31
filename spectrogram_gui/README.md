# Spectrogram Analyzer - Modern Edition

A modern, high-performance spectrogram analysis tool with enhanced UI/UX, built with PyQt5.

## 🚀 Features

### Enhanced Visualization
- **Advanced Zoom Controls**: Smooth zoom with mouse wheel, separate X/Y axis zoom (Ctrl/Shift modifiers)
- **Multiple Normalization Options**: Min-Max, Percentile, Log Scale, dB Scale
- **Colormap Selection**: Choose from 9 different colormaps (viridis, plasma, inferno, magma, jet, turbo, hot, cool, gray)
- **Real-time Crosshair & Measurements**: Toggle crosshair for precise measurements
- **Grid Toggle**: Show/hide grid for better visualization

### Modern Event Tagging System
- **Enhanced Annotation Dialog**: Add detailed annotations with type, confidence, priority, and tags
- **Inline Editing**: Edit annotations directly in the table
- **Color-Coded Tags**: Visual distinction between different event types
- **Advanced Filtering**: Filter by type, search by description/tags
- **Multiple Export Formats**: Export annotations as CSV, JSON, or Excel
- **Keyboard Shortcuts**: Quick annotation with keyboard shortcuts (A: Add, E: Edit, Del: Delete)

### Performance Optimizations
- **Multi-threaded Processing**: Background audio processing with ThreadPool
- **Smart Caching**: LRU cache for processed spectrograms
- **Batch Processing**: Process multiple files simultaneously
- **Memory Monitoring**: Real-time memory usage tracking

### Modern UI/UX
- **Material Design 3 Inspired Theme**: Clean, modern dark theme with smooth gradients
- **Responsive Layout**: Collapsible panels and dockable widgets
- **Grouped Toolbar Actions**: Organized toolbar with intuitive icons
- **Tab-based Organization**: Separate tabs for settings, annotations, detection, and analysis
- **Drag & Drop Support**: Easy file management with drag and drop

### Audio Playback
- **Variable Speed Playback**: 0.5x, 1.0x, 1.5x, 2.0x speed options
- **Volume Control**: Adjustable volume with visual feedback
- **Seek & Scrub**: Click to seek, drag to scrub through audio
- **Visual Position Marker**: Red line showing current playback position

## 📋 Requirements

- Python 3.8+
- PyQt5 5.15+
- See `requirements.txt` for full list

## 🔧 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd spectrogram_gui
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## 🎯 Usage

### Basic Workflow

1. **Load Audio Files**
   - Click the folder icon or drag & drop .wav/.flac files
   - Double-click a file to load and process it

2. **Adjust Visualization**
   - Use zoom controls or mouse wheel to zoom
   - Select normalization method from dropdown
   - Choose your preferred colormap

3. **Add Annotations**
   - Click twice on the spectrogram to mark start/end
   - Or use the "Add Annotation" button for manual entry
   - Edit annotations by double-clicking in the table

4. **Configure Analysis**
   - Adjust STFT parameters (window, size, overlap)
   - Apply filters through the filter dialog
   - Configure detection parameters

5. **Export Results**
   - Export spectrogram as image (PNG)
   - Export data as NumPy arrays (.npz)
   - Export annotations as CSV/JSON/Excel

### Keyboard Shortcuts

- **Space**: Play/Pause audio
- **A**: Add new annotation
- **E**: Edit selected annotation
- **Delete**: Remove selected annotation
- **Ctrl+C/V**: Copy/Paste annotations
- **Ctrl+O**: Open files
- **Ctrl+S**: Save session

### Mouse Controls

- **Left Click**: Seek to position / Start annotation
- **Right Click**: Context menu / End annotation
- **Mouse Wheel**: Zoom in/out
- **Ctrl + Wheel**: Zoom Y-axis only
- **Shift + Wheel**: Zoom X-axis only
- **Middle Button Drag**: Pan view

## 🎨 Customization

### Theme Customization
Edit `styles/modern_theme.qss` to customize colors and styling:
- Color palette defined at the top
- Individual component styles organized by sections

### Adding New Colormaps
Add colormap names to the combo box in `enhanced_spectrogram_canvas.py`

### Modifying Shortcuts
Edit shortcuts in `modern_event_annotator.py` `setup_shortcuts()` method

## 🔬 Advanced Features

### Batch Processing
1. Load multiple files into the file list
2. Click "Batch Process"
3. Select output directory
4. All files will be processed and saved

### Custom Filters
1. Click the filter button or use toolbar
2. Configure high-pass, low-pass, band-pass, or notch filters
3. Filters are applied in real-time

### Detection Algorithms
- 1D Doppler Detection
- 2D Doppler Detection
- Configurable parameters for each

## 🐛 Troubleshooting

### Performance Issues
- Reduce window size in STFT settings
- Clear cache if memory usage is high
- Close unused tabs

### Display Issues
- Check high DPI settings in main.py
- Try different Qt styles (Fusion recommended)

### Audio Playback Issues
- Ensure sounddevice is properly installed
- Check audio device settings

## 📊 Data Formats

### Annotation CSV Format
```csv
Start,End,Site,Pixel,Type,Description,Tags,Confidence,Priority,Snapshot
```

### Spectrogram Data (.npz)
- `freqs`: Frequency array
- `times`: Time array
- `Sxx`: Spectrogram matrix (freq × time)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- PyQt5 for the GUI framework
- pyqtgraph for high-performance plotting
- NumPy/SciPy for signal processing
- qtawesome for modern icons