from __future__ import annotations

import shutil
import urllib.error
import urllib.request

from faster_whisper import download_model

from .config import MODEL_DIR, MODEL_NAME

_PREFLIGHT_URL = f"https://huggingface.co/api/models/{MODEL_NAME}"
_PREFLIGHT_TIMEOUT = 5.0


def is_model_present() -> bool:
    return MODEL_DIR.is_dir() and (MODEL_DIR / "model.bin").exists()


def preflight_check() -> None:
    # urllib respects HTTPS_PROXY / HTTP_PROXY automatically.
    req = urllib.request.Request(
        _PREFLIGHT_URL,
        headers={"User-Agent": "transcribe-app/0.1"},
    )
    with urllib.request.urlopen(req, timeout=_PREFLIGHT_TIMEOUT) as resp:
        if resp.status != 200:
            raise urllib.error.HTTPError(
                _PREFLIGHT_URL, resp.status, f"HTTP {resp.status}", resp.headers, None
            )


def cleanup_partial_download() -> None:
    if MODEL_DIR.exists() and not is_model_present():
        shutil.rmtree(MODEL_DIR, ignore_errors=True)


def ensure_model() -> str:
    if is_model_present():
        return str(MODEL_DIR)
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    return download_model(MODEL_NAME, output_dir=str(MODEL_DIR))


def main() -> None:
    if is_model_present():
        print(f"Model already present at: {MODEL_DIR}")
        return
    print(f"Downloading {MODEL_NAME} to {MODEL_DIR} ...")
    path = ensure_model()
    print(f"Model ready at: {path}")


if __name__ == "__main__":
    main()
