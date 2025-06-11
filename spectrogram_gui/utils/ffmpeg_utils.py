import subprocess
import tempfile
import os
from pathlib import Path

# Try bundled ffmpeg first, then fall back to system-installed ffmpeg
_bundled = os.path.join(os.path.dirname(__file__), os.pardir, "resources", "ffmpeg", "ffmpeg.exe")
FFMPEG_PATH = _bundled if os.path.exists(_bundled) else "ffmpeg"

def convert_to_wav(input_path):
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"[FFMPEG] File does not exist: {input_path}")
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        wav_path = tmpfile.name

    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", str(input_path),
        str(wav_path)
    ]

    print(f"[FFMPEG] Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[FFMPEG] Converted to WAV: {wav_path}")
        return wav_path
    except subprocess.CalledProcessError as e:
        print(f"[FFMPEG] Conversion failed: {e.stderr.decode()}")
        return None
