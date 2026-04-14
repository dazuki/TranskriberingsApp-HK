from faster_whisper import download_model

from .config import MODEL_DIR, MODEL_NAME


def main() -> None:
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_NAME} to {MODEL_DIR} ...")
    path = download_model(MODEL_NAME, output_dir=str(MODEL_DIR))
    print(f"Model ready at: {path}")


if __name__ == "__main__":
    main()
