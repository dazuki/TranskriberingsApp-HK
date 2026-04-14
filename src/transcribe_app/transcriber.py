from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from faster_whisper import WhisperModel

from .config import LANGUAGE, MODEL_DIR, MODEL_NAME


class Transcriber:
    def __init__(self) -> None:
        model_source = str(MODEL_DIR) if MODEL_DIR.exists() else MODEL_NAME
        self.model = WhisperModel(model_source, device="cpu", compute_type="int8")

    def transcribe(self, audio_path: Path) -> Iterator[tuple[float, float, str]]:
        segments, _info = self.model.transcribe(
            str(audio_path),
            language=LANGUAGE,
            beam_size=5,
            vad_filter=True,
        )
        for seg in segments:
            yield seg.start, seg.end, seg.text.strip()
