import sys
from pathlib import Path


def _base_dir() -> Path:
    # When frozen by PyInstaller, models/ and transcripts/ live next to the exe.
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parents[2]


BASE_DIR = _base_dir()
MODEL_DIR = BASE_DIR / "models" / "faster-whisper-medium"
MODEL_NAME = "Systran/faster-whisper-medium"

LANGUAGE = "sv"
SAMPLE_RATE = 16000
CHANNELS = 1

TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
