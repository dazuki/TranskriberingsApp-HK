from __future__ import annotations

import datetime as dt
import shutil
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from .config import TRANSCRIPTS_DIR
from .download_model import ensure_model, is_model_present
from .recorder import Recorder, list_input_devices
from .transcriber import Transcriber


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Transkribering")
        self.root.geometry("780x560")

        self.recorder = Recorder()
        self.transcriber: Transcriber | None = None
        self.recording = False
        self.audio_path: Path | None = None
        self._devices: list[tuple[int, str]] = []

        self._build_ui()
        self._refresh_devices()
        self._ensure_model_on_startup()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        device_row = ttk.Frame(frame)
        device_row.pack(fill=tk.X)
        ttk.Label(device_row, text="Mikrofon:").pack(side=tk.LEFT)
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(
            device_row,
            textvariable=self.device_var,
            state="readonly",
            width=60,
        )
        self.device_combo.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        self.refresh_btn = ttk.Button(device_row, text="↻", width=3, command=self._refresh_devices)
        self.refresh_btn.pack(side=tk.LEFT)

        controls = ttk.Frame(frame)
        controls.pack(fill=tk.X, pady=(8, 0))

        self.record_btn = ttk.Button(controls, text="● Spela in", command=self.toggle_record)
        self.record_btn.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Redo")
        ttk.Label(controls, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        self.progress = ttk.Progressbar(controls, mode="indeterminate", length=160)
        self.progress.pack(side=tk.RIGHT)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        self.text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Segoe UI", 11))
        self.text.pack(fill=tk.BOTH, expand=True)

    def _refresh_devices(self) -> None:
        self._devices = list_input_devices()
        labels = [label for _, label in self._devices]
        self.device_combo.configure(values=labels)
        if not labels:
            self.device_var.set("")
            self.record_btn.configure(state=tk.DISABLED)
            self.status_var.set("Inga mikrofoner hittades")
            return
        self.record_btn.configure(state=tk.NORMAL)
        # Prefer the "(default)" entry, otherwise pick the first.
        default_idx = next(
            (i for i, (_, label) in enumerate(self._devices) if "(default)" in label),
            0,
        )
        self.device_combo.current(default_idx)

    def _selected_device_index(self) -> int | None:
        if not self._devices:
            return None
        i = self.device_combo.current()
        if i < 0:
            return None
        return self._devices[i][0]

    def _ensure_model_on_startup(self) -> None:
        if is_model_present():
            return
        self._set_button_to_download_mode()
        self.status_var.set("Modell saknas")

    def _set_button_to_download_mode(self) -> None:
        self.record_btn.configure(
            text="Ladda ner modell",
            state=tk.NORMAL,
            command=self._prompt_and_download_model,
        )

    def _set_button_to_record_mode(self) -> None:
        self.record_btn.configure(
            text="● Spela in",
            state=tk.NORMAL if self._devices else tk.DISABLED,
            command=self.toggle_record,
        )

    def _prompt_and_download_model(self) -> None:
        proceed = messagebox.askyesno(
            "Ladda ner transkriberingsmodell",
            (
                "Transkriberingsmodellen saknas på den här datorn och måste "
                "laddas ner innan appen kan användas.\n\n"
                "Vad som laddas ner:\n"
                "  • Modell: Systran/faster-whisper-medium\n"
                "  • Storlek: ca 1,4 GB\n"
                "  • Källa: Hugging Face (huggingface.co)\n"
                "  • Sparas lokalt i mappen 'models/' i projektet\n\n"
                "Viktigt att veta:\n"
                "  • Detta är den enda gången appen ansluter till internet.\n"
                "  • Efter nedladdning fungerar appen helt lokalt.\n"
                "  • Inget ljud och ingen transkribering lämnar din dator.\n"
                "  • Nedladdningen kan ta flera minuter beroende på din\n"
                "    internetanslutning.\n\n"
                "Vill du fortsätta med nedladdningen?"
            ),
            icon=messagebox.QUESTION,
        )
        if not proceed:
            self.status_var.set("Nedladdning avbruten - klicka på knappen för att försöka igen")
            return
        self.record_btn.configure(state=tk.DISABLED)
        self.device_combo.configure(state=tk.DISABLED)
        self.refresh_btn.configure(state=tk.DISABLED)
        self.status_var.set("Laddar ner modell (~1,4 GB)...")
        self.progress.start(12)
        threading.Thread(target=self._download_model_worker, daemon=True).start()

    def _download_model_worker(self) -> None:
        try:
            ensure_model()
        except Exception as exc:
            err = str(exc)
            self.root.after(0, self._on_model_download_error, err)
            return
        self.root.after(0, self._on_model_download_done)

    def _on_model_download_done(self) -> None:
        self.progress.stop()
        self.device_combo.configure(state="readonly")
        self.refresh_btn.configure(state=tk.NORMAL)
        self._set_button_to_record_mode()
        self.status_var.set("Modell nedladdad - redo")

    def _on_model_download_error(self, err: str) -> None:
        self.progress.stop()
        self.device_combo.configure(state="readonly")
        self.refresh_btn.configure(state=tk.NORMAL)
        self._set_button_to_download_mode()
        self.status_var.set("Kunde inte ladda ner modellen")
        messagebox.showerror(
            "Nedladdningsfel",
            "Modellen kunde inte laddas ner. Kontrollera internetanslutningen "
            "och försök igen, eller kör `uv run download-model` manuellt.\n\n"
            f"Detaljer: {err}",
        )

    def toggle_record(self) -> None:
        if not self.recording:
            self.start_recording()
        else:
            self.stop_and_transcribe()

    def start_recording(self) -> None:
        device = self._selected_device_index()
        try:
            self.recorder.start(device=device)
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Inspelningsfel", str(exc))
            return
        self.recording = True
        self.record_btn.configure(text="■ Stoppa")
        self.device_combo.configure(state=tk.DISABLED)
        self.refresh_btn.configure(state=tk.DISABLED)
        self.status_var.set("Spelar in...")
        self.text.delete("1.0", tk.END)

    def stop_and_transcribe(self) -> None:
        self.recording = False
        self.record_btn.configure(text="● Spela in", state=tk.DISABLED)
        self.status_var.set("Sparar ljud...")

        tmp_dir = Path(tempfile.gettempdir()) / "transcribe-app"
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        wav_path = tmp_dir / f"recording-{timestamp}.wav"

        try:
            self.audio_path = self.recorder.stop(wav_path)
        except Exception as exc:
            messagebox.showerror("Inspelningsfel", str(exc))
            self.record_btn.configure(state=tk.NORMAL)
            self.device_combo.configure(state="readonly")
            self.refresh_btn.configure(state=tk.NORMAL)
            self.status_var.set("Redo")
            return

        self.status_var.set("Transkriberar (detta kan ta en stund)...")
        self.progress.start(12)
        threading.Thread(
            target=self._run_transcription, args=(self.audio_path,), daemon=True
        ).start()

    def _run_transcription(self, audio_path: Path) -> None:
        try:
            if self.transcriber is None:
                self.root.after(0, lambda: self.status_var.set("Laddar modell..."))
                self.transcriber = Transcriber()
                self.root.after(0, lambda: self.status_var.set("Transkriberar..."))

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
        # Keep a copy of the WAV next to the transcript for debugging/playback.
        wav_copy = TRANSCRIPTS_DIR / audio_path.name
        try:
            shutil.copy2(audio_path, wav_copy)
        except OSError as exc:
            print(f"[ui] failed to copy WAV to transcripts dir: {exc}")
        return out_path

    def _on_done(self, out_path: Path) -> None:
        self.progress.stop()
        self.record_btn.configure(state=tk.NORMAL)
        self.device_combo.configure(state="readonly")
        self.refresh_btn.configure(state=tk.NORMAL)
        self.status_var.set(f"Sparad: {out_path}")

    def _on_error(self, err: str) -> None:
        self.progress.stop()
        self.record_btn.configure(state=tk.NORMAL)
        self.device_combo.configure(state="readonly")
        self.refresh_btn.configure(state=tk.NORMAL)
        self.status_var.set("Fel")
        messagebox.showerror("Transkriberingsfel", err)


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def run() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()
