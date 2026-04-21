"""Build the distributable: PyInstaller onedir + bundled model + .7z archive.

Usage:
    uv run --with pyinstaller python build.py            # includes model
    uv run --with pyinstaller python build.py --no-model # skips model
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_SRC = ROOT / "models" / "faster-whisper-medium"
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"
APP_DIR = DIST_DIR / "transcribe-app"
SPEC_FILE = ROOT / "transcribe-app.spec"


def read_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def run_pyinstaller(step: str) -> None:
    print(f"[{step}] Running PyInstaller...")
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    if APP_DIR.exists():
        shutil.rmtree(APP_DIR)
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", str(SPEC_FILE), "--noconfirm"],
        check=True,
        cwd=ROOT,
    )


def copy_model(step: str) -> None:
    print(f"[{step}] Copying model into dist...")
    if not MODEL_SRC.exists():
        sys.exit(
            f"ERROR: model not found at {MODEL_SRC}\n"
            "Run `uv run download-model` first, or use --no-model."
        )
    dest = APP_DIR / "models" / "faster-whisper-medium"
    shutil.copytree(MODEL_SRC, dest)


def make_archive(version: str, step: str, no_model: bool) -> Path:
    suffix = "-nomodel" if no_model else ""
    archive = DIST_DIR / f"transcribe-app-{version}{suffix}.7z"
    if archive.exists():
        archive.unlink()
    print(f"[{step}] Compressing to {archive.name} (7zr, LZMA2 fastest, multi-threaded)...")
    sevenzr = shutil.which("7zr") or shutil.which("7z")
    if not sevenzr:
        sys.exit("ERROR: 7zr not found on PATH. Install 7-Zip or add 7zr.exe to ~/.local/bin.")
    subprocess.run(
        [
            sevenzr,
            "a",
            "-t7z",
            "-m0=lzma2",
            "-mx=1",
            "-mmt=on",
            str(archive),
            str(APP_DIR),
        ],
        check=True,
        cwd=ROOT,
    )
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Build transcribe-app distributable.")
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip bundling the Whisper model (produces a smaller archive).",
    )
    args = parser.parse_args()

    total = 2 if args.no_model else 3
    version = read_version()
    run_pyinstaller(f"1/{total}")
    if args.no_model:
        print("Skipping model copy (--no-model).")
    else:
        copy_model(f"2/{total}")
    archive = make_archive(version, f"{total}/{total}", args.no_model)
    size_mb = archive.stat().st_size / (1024 * 1024)
    print(f"\nDone: {archive} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
