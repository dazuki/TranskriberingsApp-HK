from __future__ import annotations

import queue
import threading
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from .config import CHANNELS, SAMPLE_RATE

# WDM-KS is blocked because it surfaces speakers as loopback-capable inputs.
_HOSTAPI_PRIORITY = ("Windows DirectSound", "Windows WASAPI", "MME", "Core Audio", "ALSA")
_HOSTAPI_BLOCKLIST = ("Windows WDM-KS",)

_OUTPUT_NAME_HINTS = ("output", "högtalare", "hogtalare", "speaker", "stereo mix", "stereomix")

_WRAPPER_NAME_HINTS = (
    "microsoft sound mapper",
    "primär drivrutin för ljudinfångst",
    "primar drivrutin for ljudinfangst",
    "primary sound capture driver",
)


def _is_usable_input(device_index: int, channels: int) -> tuple[bool, float | None]:
    target_rate = SAMPLE_RATE
    try:
        sd.check_input_settings(
            device=device_index, channels=channels, samplerate=target_rate, dtype="int16"
        )
        return True, target_rate
    except Exception:
        pass
    # WASAPI is strict — fall back to the device's native rate.
    try:
        info = sd.query_devices(device_index)
        native = float(info["default_samplerate"])
        sd.check_input_settings(
            device=device_index, channels=channels, samplerate=native, dtype="int16"
        )
        return True, native
    except Exception:
        return False, None


def _same_device(a: str, b: str) -> bool:
    # MME truncates device names to 31 chars, so prefix-match.
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return len(shorter) >= 20 and longer.startswith(shorter)


def list_input_devices(
    sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS
) -> list[tuple[int, str]]:
    devices = sd.query_devices()
    try:
        default_in = sd.default.device[0]
        default_name = devices[default_in]["name"] if default_in is not None else None
    except (TypeError, IndexError):
        default_name = None

    candidates: list[tuple[int, int, str, str]] = []
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) <= 0:
            continue
        name = dev["name"].strip()
        lowered = name.lower()
        if any(hint in lowered for hint in _OUTPUT_NAME_HINTS):
            continue
        if any(hint in lowered for hint in _WRAPPER_NAME_HINTS):
            continue
        host = sd.query_hostapis(dev["hostapi"])["name"]
        if host in _HOSTAPI_BLOCKLIST:
            continue
        usable, _ = _is_usable_input(idx, channels)
        if not usable:
            continue
        priority = (
            _HOSTAPI_PRIORITY.index(host) if host in _HOSTAPI_PRIORITY else len(_HOSTAPI_PRIORITY)
        )
        candidates.append((priority, idx, name, host))

    candidates.sort(key=lambda c: c[0])

    kept: list[tuple[int, str, str]] = []
    for _priority, idx, name, host in candidates:
        if any(_same_device(name, k_name) for _, k_name, _ in kept):
            continue
        kept.append((idx, name, host))

    items: list[tuple[int, str]] = []
    for idx, name, host in kept:
        is_default = default_name is not None and _same_device(name, default_name)
        marker = " (default)" if is_default else ""
        items.append((idx, f"{name} - {host}{marker}"))
    items.sort(key=lambda item: (" (default)" not in item[1], item[1].lower()))
    return items


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

    def start(self, device: int | None = None) -> None:
        if self._stream is not None:
            raise RuntimeError("Recorder already running")
        self._chunks = []
        self._stop_event.clear()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            device=device,
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

        while not self._queue.empty():
            self._chunks.append(self._queue.get_nowait())

        if not self._chunks:
            raise RuntimeError("No audio captured")

        audio = np.concatenate(self._chunks, axis=0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())
        return output_path
