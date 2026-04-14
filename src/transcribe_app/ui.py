from __future__ import annotations

import datetime as dt
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from .config import TRANSCRIPTS_DIR
from .recorder import Recorder
from .transcriber import Transcriber


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Transkribering (Svenska)")
        self.root.geometry("780x560")

        self.recorder = Recorder()
        self.transcriber: Transcriber | None = None
        self.recording = False
        self.audio_path: Path | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(frame)
        controls.pack(fill=tk.X)

        self.record_btn = ttk.Button(controls, text="● Record", command=self.toggle_record)
        self.record_btn.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(controls, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        self.progress = ttk.Progressbar(controls, mode="indeterminate", length=160)
        self.progress.pack(side=tk.RIGHT)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        self.text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Segoe UI", 11))
        self.text.pack(fill=tk.BOTH, expand=True)

    def toggle_record(self) -> None:
        if not self.recording:
            self.start_recording()
        else:
            self.stop_and_transcribe()

    def start_recording(self) -> None:
        try:
            self.recorder.start()
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Recorder error", str(exc))
            return
        self.recording = True
        self.record_btn.configure(text="■ Stop")
        self.status_var.set("Recording...")
        self.text.delete("1.0", tk.END)

    def stop_and_transcribe(self) -> None:
        self.recording = False
        self.record_btn.configure(text="● Record", state=tk.DISABLED)
        self.status_var.set("Saving audio...")

        tmp_dir = Path(tempfile.gettempdir()) / "transcribe-app"
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        wav_path = tmp_dir / f"recording-{timestamp}.wav"

        try:
            self.audio_path = self.recorder.stop(wav_path)
        except Exception as exc:
            messagebox.showerror("Recorder error", str(exc))
            self.record_btn.configure(state=tk.NORMAL)
            self.status_var.set("Ready")
            return

        self.status_var.set("Transcribing (this may take a while)...")
        self.progress.start(12)
        threading.Thread(target=self._run_transcription, args=(self.audio_path,), daemon=True).start()

    def _run_transcription(self, audio_path: Path) -> None:
        try:
            if self.transcriber is None:
                self.root.after(0, lambda: self.status_var.set("Loading model..."))
                self.transcriber = Transcriber()
                self.root.after(0, lambda: self.status_var.set("Transcribing..."))

            lines: list[str] = []
            for start, end, text in self.transcriber.transcribe(audio_path):
                line = f"[{_fmt(start)} -> {_fmt(end)}] {text}"
                lines.append(line)
                self.root.after(0, self._append_line, line)

            full_text = "\n".join(lines)
            out_path = self._save_transcript(audio_path, full_text)
            self.root.after(0, self._on_done, out_path)
        except Exception as exc:
            err = str(exc)
            self.root.after(0, self._on_error, err)

    def _append_line(self, line: str) -> None:
        self.text.insert(tk.END, line + "\n")
        self.text.see(tk.END)

    def _save_transcript(self, audio_path: Path, content: str) -> Path:
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = TRANSCRIPTS_DIR / (audio_path.stem + ".txt")
        out_path.write_text(content, encoding="utf-8")
        return out_path

    def _on_done(self, out_path: Path) -> None:
        self.progress.stop()
        self.record_btn.configure(state=tk.NORMAL)
        self.status_var.set(f"Saved: {out_path}")

    def _on_error(self, err: str) -> None:
        self.progress.stop()
        self.record_btn.configure(state=tk.NORMAL)
        self.status_var.set("Error")
        messagebox.showerror("Transcription error", err)


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def run() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()
