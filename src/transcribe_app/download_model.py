from __future__ import annotations

from faster_whisper import download_model

from .config import MODEL_DIR, MODEL_NAME


def is_model_present() -> bool:
    """True if MODEL_DIR contains a usable faster-whisper model."""
    return MODEL_DIR.is_dir() and (MODEL_DIR / "model.bin").exists()


def ensure_model() -> str:
    """Download the model to MODEL_DIR if not already present. Returns the path."""
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
