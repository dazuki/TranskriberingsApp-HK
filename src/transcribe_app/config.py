from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "faster-whisper-medium"
MODEL_NAME = "Systran/faster-whisper-medium"

LANGUAGE = "sv"
SAMPLE_RATE = 16000
CHANNELS = 1

TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
