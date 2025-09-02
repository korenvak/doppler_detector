#!/bin/bash

# Modern Spectrogram GUI Installation Script
# ==========================================

echo "======================================"
echo "Modern Spectrogram GUI Installer"
echo "======================================"
echo

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "Error: Python 3.8 or higher is required (found $python_version)"
    exit 1
fi

echo "✓ Python version: $python_version"

# Create virtual environment
echo
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo
echo "Installing dependencies..."
pip install -r requirements_new.txt

# Additional optional dependencies for better performance
echo
echo "Installing optional performance dependencies..."
pip install --no-deps accelerate 2>/dev/null || true

# Create desktop entry (Linux)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo
    echo "Creating desktop entry..."
    cat > ~/.local/share/applications/spectrogram-analyzer.desktop <<EOF
[Desktop Entry]
Name=Spectrogram Analyzer
Comment=Modern audio analysis tool with glassmorphic UI
Exec=bash -c "cd $(pwd) && source venv/bin/activate && python main_final.py"
Icon=$(pwd)/icon.png
Terminal=false
Type=Application
Categories=Audio;AudioVideo;Science;
EOF
    echo "✓ Desktop entry created"
fi

# Create launch script
echo
echo "Creating launch script..."
cat > launch.sh <<'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python main_final.py "$@"
EOF
chmod +x launch.sh

echo
echo "======================================"
echo "Installation Complete!"
echo "======================================"
echo
echo "To run the application:"
echo "  ./launch.sh"
echo
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python main_final.py"
echo
echo "Features:"
echo "  • Modern glassmorphic UI design"
echo "  • High-performance spectrogram rendering"
echo "  • Smooth zoom with FLAC boundaries"
echo "  • Real-time audio playback"
echo "  • Advanced filtering options"
echo
echo "Enjoy your enhanced audio analysis experience!"