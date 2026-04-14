from __future__ import annotations

import queue
import threading
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from .config import CHANNELS, SAMPLE_RATE


class Recorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []
        self._collector: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            print(f"[recorder] {status}")
        self._queue.put(indata.copy())

    def _collect(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self._chunks.append(chunk)

    def start(self) -> None:
        if self._stream is not None:
            raise RuntimeError("Recorder already running")
        self._chunks = []
        self._stop_event.clear()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()
        self._collector = threading.Thread(target=self._collect, daemon=True)
        self._collector.start()

    def stop(self, output_path: Path) -> Path:
        if self._stream is None:
            raise RuntimeError("Recorder not running")
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._stop_event.set()
        if self._collector is not None:
            self._collector.join()
            self._collector = None

        # Drain any leftover queued frames.
        while not self._queue.empty():
            self._chunks.append(self._queue.get_nowait())

        if not self._chunks:
            raise RuntimeError("No audio captured")

        audio = np.concatenate(self._chunks, axis=0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # int16
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())
        return output_path
