from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from faster_whisper import WhisperModel

from .config import LANGUAGE, MODEL_DIR
from .download_model import is_model_present


class Transcriber:
    def __init__(self) -> None:
        if not is_model_present():
            raise FileNotFoundError(
                f"Model not found at {MODEL_DIR}. Run `uv run download-model` first."
            )
        self.model = WhisperModel(str(MODEL_DIR), device="cpu", compute_type="int8")

    def transcribe(self, audio_path: Path) -> tuple[float, Iterator[tuple[float, float, str]]]:
        # Lower VAD threshold keeps quiet speech that Intel Smart Sound noise
        # suppression drops below the default 0.5 floor.
        segments, info = self.model.transcribe(
            str(audio_path),
            language=LANGUAGE,
            beam_size=5,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.3,
                "speech_pad_ms": 800,
                "min_silence_duration_ms": 1000,
            },
            condition_on_previous_text=False,
        )

        def _iter() -> Iterator[tuple[float, float, str]]:
            for seg in segments:
                yield seg.start, seg.end, seg.text.strip()

        return info.duration, _iter()
