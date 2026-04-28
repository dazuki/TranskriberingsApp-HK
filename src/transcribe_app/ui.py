from __future__ import annotations

import datetime as dt
import re
import shutil
import tempfile
import threading
import time
import tkinter as tk
import wave
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import numpy as np
import sounddevice as sd

from .config import TRANSCRIPTS_DIR
from .download_model import (
    cleanup_partial_download,
    ensure_model,
    is_model_present,
    preflight_check,
)
from .recorder import Recorder, list_input_devices
from .transcriber import Transcriber


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Transkribering")
        self.root.geometry("1280x700")

        self.recorder = Recorder()
        self.transcriber: Transcriber | None = None
        self.recording = False
        self.audio_path: Path | None = None
        self._devices: list[tuple[int, str]] = []
        self._wav_paths: list[Path] = []
        self._playback_audio: np.ndarray | None = None
        self._playback_sr: int = 16000
        self._playback_duration: float = 0.0
        self._playback_elapsed: float = 0.0
        self._playback_start: float | None = None
        self._playback_paused: bool = False
        self._playback_job: str | None = None
        self._seek_dragging: bool = False

        self._build_ui()
        self._refresh_devices()
        self._refresh_file_list()
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

        self.file_btn = ttk.Button(
            controls,
            text="Välj ljudfil för transkribering...",
            command=self.pick_file_and_transcribe,
        )
        self.file_btn.pack(side=tk.LEFT, padx=(6, 0))

        self.status_var = tk.StringVar(value="Redo")
        ttk.Label(controls, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        self.spinner_label = ttk.Label(controls, text="", font=("Consolas", 14), width=2)
        self.spinner_label.pack(side=tk.RIGHT)
        self._spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._spinner_idx = 0
        self._spinner_job: str | None = None

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        left_frame = ttk.Frame(frame, width=320)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        left_frame.pack_propagate(False)

        ttk.Separator(frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        right_frame = ttk.Frame(frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        list_header = ttk.Frame(left_frame)
        list_header.pack(fill=tk.X)
        ttk.Label(list_header, text="Inspelningar").pack(side=tk.LEFT)
        ttk.Button(list_header, text="↻", width=3, command=self._refresh_file_list).pack(
            side=tk.RIGHT
        )

        list_inner = ttk.Frame(left_frame)
        list_inner.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        _sb = ttk.Scrollbar(list_inner, orient=tk.VERTICAL)
        self.file_list = tk.Listbox(
            list_inner,
            yscrollcommand=_sb.set,
            selectmode=tk.SINGLE,
            activestyle="dotbox",
            font=("Segoe UI", 9),
        )
        _sb.configure(command=self.file_list.yview)
        _sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.file_list.bind("<<ListboxSelect>>", self._on_file_select)

        play_row = ttk.Frame(left_frame)
        play_row.pack(fill=tk.X, pady=(4, 0))
        self.play_btn = ttk.Button(
            play_row, text="▶ Spela upp", command=self._play_selected, state=tk.DISABLED
        )
        self.play_btn.pack(side=tk.LEFT)
        self.delete_btn = ttk.Button(
            play_row, text="Ta bort", command=self._delete_selected, state=tk.DISABLED
        )
        self.delete_btn.pack(side=tk.LEFT, padx=(6, 0))

        self.text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, font=("Segoe UI", 11))
        self.text.pack(fill=tk.BOTH, expand=True)
        self.text.tag_configure("current_line", background="#fff3b0")

        seek_frame = ttk.Frame(right_frame)
        seek_frame.pack(fill=tk.X, pady=(4, 0))
        self.seek_var = tk.DoubleVar(value=0.0)
        self.seek_scale = ttk.Scale(
            seek_frame,
            orient=tk.HORIZONTAL,
            variable=self.seek_var,
            from_=0,
            to=100,
            command=self._on_seek_drag,
        )
        self.seek_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.seek_scale.bind("<ButtonPress-1>", self._on_seek_press)
        self.seek_scale.bind("<ButtonRelease-1>", self._on_seek_release)
        self.time_label = ttk.Label(
            seek_frame, text="", font=("Segoe UI", 9), width=13, anchor=tk.E
        )
        self.time_label.pack(side=tk.RIGHT)

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
        self.file_btn.configure(state=tk.DISABLED)

    def _set_button_to_record_mode(self) -> None:
        self.record_btn.configure(
            text="● Spela in",
            state=tk.NORMAL if self._devices else tk.DISABLED,
            command=self.toggle_record,
        )
        self.file_btn.configure(state=tk.NORMAL)

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
        self.file_btn.configure(state=tk.DISABLED)
        self.device_combo.configure(state=tk.DISABLED)
        self.refresh_btn.configure(state=tk.DISABLED)
        self.status_var.set("Laddar ner modell (~1,4 GB)...")
        self._spinner_start()
        threading.Thread(target=self._download_model_worker, daemon=True).start()

    def _download_model_worker(self) -> None:
        try:
            preflight_check()
            ensure_model()
        except Exception as exc:
            cleanup_partial_download()
            title, body = _classify_download_error(exc)
            self.root.after(0, self._on_model_download_error, title, body)
            return
        self.root.after(0, self._on_model_download_done)

    def _on_model_download_done(self) -> None:
        self._spinner_stop()
        self.device_combo.configure(state="readonly")
        self.refresh_btn.configure(state=tk.NORMAL)
        self._set_button_to_record_mode()
        self.status_var.set("Modell nedladdad - redo")

    def _on_model_download_error(self, title: str, body: str) -> None:
        self._spinner_stop()
        self.device_combo.configure(state="readonly")
        self.refresh_btn.configure(state=tk.NORMAL)
        self._set_button_to_download_mode()
        self.status_var.set("Kunde inte ladda ner modellen")
        messagebox.showerror(title, body)

    def toggle_record(self) -> None:
        if not self.recording:
            self.start_recording()
        else:
            self.stop_and_transcribe()

    def start_recording(self) -> None:
        self._stop_playback()
        device = self._selected_device_index()
        try:
            self.recorder.start(device=device)
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Inspelningsfel", str(exc))
            return
        self.recording = True
        self.record_btn.configure(text="■ Stoppa")
        self.file_btn.configure(state=tk.DISABLED)
        self.device_combo.configure(state=tk.DISABLED)
        self.refresh_btn.configure(state=tk.DISABLED)
        self.play_btn.configure(state=tk.DISABLED)
        self.delete_btn.configure(state=tk.DISABLED)
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
            self.file_btn.configure(state=tk.NORMAL)
            self.device_combo.configure(state="readonly")
            self.refresh_btn.configure(state=tk.NORMAL)
            self.status_var.set("Redo")
            return

        transcribe_now = messagebox.askyesno(
            "Transkribera inspelningen?",
            (
                "Inspelningen är klar.\n\n"
                "Vill du transkribera den nu?\n\n"
                "Om du svarar Nej sparas ljudfilen och du kan transkribera "
                "den senare via 'Välj ljudfil för transkribering...'."
            ),
            icon=messagebox.QUESTION,
        )

        if not transcribe_now:
            saved_path = self._keep_recording(self.audio_path)
            self.record_btn.configure(state=tk.NORMAL)
            self.file_btn.configure(state=tk.NORMAL)
            self.device_combo.configure(state="readonly")
            self.refresh_btn.configure(state=tk.NORMAL)
            self._refresh_file_list()
            self.status_var.set(f"Ljudfil sparad: {saved_path}")
            return

        self._show_transcription_placeholder()
        self.play_btn.configure(state=tk.DISABLED)
        self.delete_btn.configure(state=tk.DISABLED)
        self.status_var.set("Transkriberar (detta kan ta en stund)...")
        self._spinner_start()
        threading.Thread(
            target=self._run_transcription, args=(self.audio_path, True), daemon=True
        ).start()

    def _keep_recording(self, wav_path: Path) -> Path:
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        dest = TRANSCRIPTS_DIR / wav_path.name
        try:
            shutil.copy2(wav_path, dest)
        except OSError as exc:
            print(f"[ui] failed to copy WAV to transcripts dir: {exc}")
            return wav_path
        return dest

    def pick_file_and_transcribe(self) -> None:
        path_str = filedialog.askopenfilename(
            title="Välj ljudfil",
            filetypes=[
                ("Ljudfiler", "*.wav *.mp3 *.m4a *.mp4 *.ogg *.oga *.flac *.webm *.aac *.wma"),
                ("Alla filer", "*.*"),
            ],
        )
        if not path_str:
            return
        audio_path = Path(path_str)
        if not audio_path.is_file():
            messagebox.showerror("Filfel", f"Filen kunde inte läsas: {audio_path}")
            return

        self.audio_path = audio_path
        self.record_btn.configure(state=tk.DISABLED)
        self.file_btn.configure(state=tk.DISABLED)
        self.device_combo.configure(state=tk.DISABLED)
        self.refresh_btn.configure(state=tk.DISABLED)
        self.play_btn.configure(state=tk.DISABLED)
        self.delete_btn.configure(state=tk.DISABLED)
        self._show_transcription_placeholder()
        self.status_var.set("Transkriberar (detta kan ta en stund)...")
        self._spinner_start()
        threading.Thread(
            target=self._run_transcription, args=(audio_path, True), daemon=True
        ).start()

    def _run_transcription(self, audio_path: Path, copy_audio: bool) -> None:
        try:
            if self.transcriber is None:
                self.root.after(0, lambda: self.status_var.set("Laddar modell..."))
                self.transcriber = Transcriber()
            self.root.after(0, lambda: self.status_var.set("Analyserar ljud..."))

            duration, segments = self.transcriber.transcribe(audio_path)
            self.root.after(0, self._on_transcription_started, duration)

            lines: list[str] = []
            for start, end, text in segments:
                line = f"[{_fmt(start)} -> {_fmt(end)}] {text}"
                lines.append(line)
                pct = min(100, int((end / duration) * 100)) if duration > 0 else 0
                self.root.after(0, self._append_line_with_progress, line, pct)

            full_text = "\n".join(lines)
            out_path = self._save_transcript(audio_path, full_text, copy_audio=copy_audio)
            self.root.after(0, self._on_done, out_path)
        except Exception as exc:
            err = str(exc)
            self.root.after(0, self._on_error, err)

    def _show_transcription_placeholder(self) -> None:
        self.text.delete("1.0", tk.END)
        self.text.insert(
            "1.0",
            "Förbereder transkribering, vänta...\n"
            "(Modellen laddas och ljudet analyseras innan text visas här.)\n",
        )

    def _on_transcription_started(self, duration: float) -> None:
        self.text.delete("1.0", tk.END)
        self.status_var.set(f"Transkriberar 0% (0:00 / {_fmt(duration)})")

    def _append_line_with_progress(self, line: str, pct: int) -> None:
        self.text.insert(tk.END, line + "\n")
        self.text.see(tk.END)
        self.status_var.set(f"Transkriberar {pct}%")

    def _save_transcript(self, audio_path: Path, content: str, *, copy_audio: bool) -> Path:
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = TRANSCRIPTS_DIR / (audio_path.stem + ".txt")
        out_path.write_text(content, encoding="utf-8")
        if copy_audio:
            wav_copy = TRANSCRIPTS_DIR / audio_path.name
            if wav_copy != audio_path:
                try:
                    shutil.copy2(audio_path, wav_copy)
                except OSError as exc:
                    print(f"[ui] failed to copy WAV to transcripts dir: {exc}")
        return out_path

    def _refresh_file_list(self) -> None:
        self._wav_paths: list[Path] = []
        if TRANSCRIPTS_DIR.exists():
            self._wav_paths = sorted(
                TRANSCRIPTS_DIR.glob("*.wav"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        self.file_list.delete(0, tk.END)
        for p in self._wav_paths:
            marker = "" if (TRANSCRIPTS_DIR / (p.stem + ".txt")).exists() else " *"
            self.file_list.insert(tk.END, p.stem + marker)

    def _on_file_select(self, _event: object) -> None:
        self._stop_playback()
        sel = self.file_list.curselection()
        if not sel:
            return
        wav_path = self._wav_paths[sel[0]]
        txt_path = TRANSCRIPTS_DIR / (wav_path.stem + ".txt")
        self.text.delete("1.0", tk.END)
        if txt_path.exists():
            self.text.insert("1.0", txt_path.read_text(encoding="utf-8"))
            self.status_var.set(f"Visar: {txt_path.name}")
        else:
            self.status_var.set(f"Ingen transkription hittad for {wav_path.name}")
        self.play_btn.configure(state=tk.NORMAL)
        self.delete_btn.configure(state=tk.NORMAL)

    def _delete_selected(self) -> None:
        sel = self.file_list.curselection()
        if not sel:
            return
        wav_path = self._wav_paths[sel[0]]
        txt_path = TRANSCRIPTS_DIR / (wav_path.stem + ".txt")
        if not messagebox.askyesno(
            "Ta bort inspelning",
            f"Ta bort '{wav_path.stem}' och tillhörande transkription?\n\nDetta kan inte ångras.",
            icon=messagebox.WARNING,
        ):
            return
        self._stop_playback()
        for path in (wav_path, txt_path):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                messagebox.showerror("Fel", f"Kunde inte ta bort {path.name}: {exc}")
                return
        self.text.delete("1.0", tk.END)
        self.play_btn.configure(state=tk.DISABLED)
        self.delete_btn.configure(state=tk.DISABLED)
        self.status_var.set(f"Borttagen: {wav_path.stem}")
        self._refresh_file_list()

    def _play_selected(self) -> None:
        if self._playback_paused:
            self._start_playback_from(self._playback_elapsed)
            self.play_btn.configure(text="⏸ Pausa")
            self._playback_tick()
            return
        if self._playback_start is not None:
            # Pause
            sd.stop()
            if self._playback_job is not None:
                self.root.after_cancel(self._playback_job)
                self._playback_job = None
            self._playback_elapsed = self._current_pos()
            self._playback_start = None
            self._playback_paused = True
            self.play_btn.configure(text="▶ Fortsätt")
            return
        # Fresh load
        sel = self.file_list.curselection()
        if not sel:
            return
        wav_path = self._wav_paths[sel[0]]
        try:
            with wave.open(str(wav_path), "rb") as wf:
                n_frames = wf.getnframes()
                sr = wf.getframerate()
                nc = wf.getnchannels()
                sw = wf.getsampwidth()
                frames = wf.readframes(n_frames)
        except Exception as exc:
            self.status_var.set(f"Uppspelningsfel: {exc}")
            return
        dtype = np.int16 if sw == 2 else np.int32
        audio = np.frombuffer(frames, dtype=dtype)
        if nc > 1:
            audio = audio.reshape(-1, nc)
        self._playback_audio = audio
        self._playback_sr = sr
        self._playback_duration = n_frames / sr
        self._playback_elapsed = 0.0
        self.seek_scale.configure(to=self._playback_duration)
        self.seek_var.set(0)
        self._start_playback_from(0.0)
        self.play_btn.configure(text="⏸ Pausa")
        self._playback_tick()

    def _start_playback_from(self, pos: float) -> None:
        if self._playback_audio is None:
            return
        sample_offset = int(pos * self._playback_sr)
        audio = self._playback_audio
        sliced = audio[sample_offset:] if audio.ndim == 1 else audio[sample_offset:, :]
        sd.play(sliced, samplerate=self._playback_sr)
        self._playback_elapsed = pos
        self._playback_start = time.monotonic()
        self._playback_paused = False

    def _current_pos(self) -> float:
        if self._playback_start is None:
            return self._playback_elapsed
        return self._playback_elapsed + (time.monotonic() - self._playback_start)

    def _stop_playback(self) -> None:
        sd.stop()
        if self._playback_job is not None:
            self.root.after_cancel(self._playback_job)
            self._playback_job = None
        self._playback_audio = None
        self._playback_start = None
        self._playback_elapsed = 0.0
        self._playback_paused = False
        self.time_label.configure(text="")
        self.seek_var.set(0)
        self.text.tag_remove("current_line", "1.0", tk.END)
        self.play_btn.configure(text="▶ Spela upp")

    def _playback_tick(self) -> None:
        if self._playback_start is None:
            return
        pos = self._current_pos()
        if pos >= self._playback_duration:
            self._stop_playback()
            return
        if not self._seek_dragging:
            self.seek_var.set(pos)
        self.time_label.configure(text=f"{_fmt(pos)} / {_fmt(self._playback_duration)}")
        self._highlight_current_line(pos)
        self._playback_job = self.root.after(100, self._playback_tick)

    def _on_seek_drag(self, value: str) -> None:
        if self._seek_dragging:
            self.time_label.configure(
                text=f"{_fmt(float(value))} / {_fmt(self._playback_duration)}"
            )

    def _on_seek_press(self, _event: object) -> None:
        self._seek_dragging = True

    def _on_seek_release(self, _event: object) -> None:
        self._seek_dragging = False
        pos = min(self.seek_var.get(), self._playback_duration)
        was_playing = self._playback_start is not None
        sd.stop()
        if self._playback_job is not None:
            self.root.after_cancel(self._playback_job)
            self._playback_job = None
        self._playback_start = None
        self._playback_elapsed = pos
        if was_playing:
            self._start_playback_from(pos)
            self.play_btn.configure(text="⏸ Pausa")
            self._playback_tick()

    def _highlight_current_line(self, elapsed: float) -> None:
        self.text.tag_remove("current_line", "1.0", tk.END)
        content = self.text.get("1.0", tk.END)
        for i, line in enumerate(content.splitlines(), start=1):
            m = re.match(r"\[(\d+:\d+) -> (\d+:\d+)\]", line)
            if not m:
                continue
            sm, ss = m.group(1).split(":")
            em, es = m.group(2).split(":")
            start = int(sm) * 60 + int(ss)
            end = int(em) * 60 + int(es)
            if start <= elapsed < end:
                self.text.tag_add("current_line", f"{i}.0", f"{i}.end")
                self.text.see(f"{i}.0")
                return

    def _on_done(self, out_path: Path) -> None:
        self._spinner_stop()
        self.record_btn.configure(state=tk.NORMAL)
        self.file_btn.configure(state=tk.NORMAL)
        self.device_combo.configure(state="readonly")
        self.refresh_btn.configure(state=tk.NORMAL)
        self._refresh_file_list()
        self.status_var.set(f"Sparad: {out_path}")

    def _on_error(self, err: str) -> None:
        self._spinner_stop()
        self.record_btn.configure(state=tk.NORMAL)
        self.file_btn.configure(state=tk.NORMAL)
        self.device_combo.configure(state="readonly")
        self.refresh_btn.configure(state=tk.NORMAL)
        self.status_var.set("Fel")
        messagebox.showerror("Transkriberingsfel", err)

    def _spinner_start(self) -> None:
        if self._spinner_job is not None:
            return
        self._spinner_idx = 0
        self._spinner_tick()

    def _spinner_stop(self) -> None:
        if self._spinner_job is not None:
            self.root.after_cancel(self._spinner_job)
            self._spinner_job = None
        self.spinner_label.configure(text="")

    def _spinner_tick(self) -> None:
        frame = self._spinner_frames[self._spinner_idx % len(self._spinner_frames)]
        self.spinner_label.configure(text=frame)
        self._spinner_idx += 1
        self._spinner_job = self.root.after(80, self._spinner_tick)


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _classify_download_error(exc: Exception) -> tuple[str, str]:
    msg = str(exc).lower()
    raw = str(exc)
    proxy_hint = (
        "\n\nTips: Om du är på kommunens nätverk kan en proxy krävas. "
        "Kontrollera att miljövariablerna HTTPS_PROXY och HTTP_PROXY "
        "är korrekt inställda innan du startar appen."
    )

    if "proxy" in msg:
        return (
            "Proxyfel",
            "Proxyservern kunde inte nås eller svarade med ett fel. "
            "Kontrollera miljövariablerna HTTPS_PROXY och HTTP_PROXY, "
            "eller kontakta IT-avdelningen."
            f"\n\nDetaljer: {raw}",
        )

    if "ssl" in msg or "cert" in msg or "certificate" in msg:
        return (
            "SSL- eller certifikatfel",
            "Det uppstod ett SSL- eller certifikatfel vid anslutning till "
            "huggingface.co. Detta händer ofta när kommunens proxy gör "
            "SSL-inspection. Kontakta IT-avdelningen eller kontrollera att "
            "rätt rot-certifikat är installerat."
            f"\n\nDetaljer: {raw}",
        )

    if any(
        token in msg
        for token in (
            "name or service not known",
            "nameresolutionerror",
            "temporary failure in name resolution",
            "getaddrinfo",
            "connection refused",
            "connection reset",
            "no route",
            "unreachable",
            "timed out",
            "timeout",
            "urlopen error",
            "connection error",
        )
    ):
        return (
            "Ingen anslutning",
            "Kunde inte ansluta till huggingface.co. Kontrollera att datorn "
            "har internetanslutning och försök igen." + proxy_hint + f"\n\nDetaljer: {raw}",
        )

    if isinstance(exc, OSError):
        return (
            "Diskfel",
            "Kunde inte skriva modellfiler till disk. Kontrollera att det "
            "finns minst 2 GB ledigt utrymme och att du har skrivrättigheter "
            "till projektmappen."
            f"\n\nDetaljer: {raw}",
        )

    return (
        "Nedladdningsfel",
        "Modellen kunde inte laddas ner. Försök igen, eller kör "
        "`uv run download-model` manuellt från ett terminalfönster."
        + proxy_hint
        + f"\n\nDetaljer: {raw}",
    )


def run() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()
