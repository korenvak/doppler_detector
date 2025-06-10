import subprocess
import tempfile
import os
from pathlib import Path

FFMPEG_PATH = os.path.join("resources", "ffmpeg", "ffmpeg.exe")

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
