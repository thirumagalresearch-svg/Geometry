import argparse
from pathlib import Path

from scripts.config import setup_env
from scripts.mediapipe_landmarks import extract_landmarks
from scripts.utils import load_image


def main() -> None:
    """Run a simple MediaPipe landmark extraction test."""
    setup_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input selfie image.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = load_image(image_path)
    landmarks = extract_landmarks(image)
    if not landmarks:
        raise RuntimeError("No face detected.")

    print("Detected landmarks:", landmarks)


if __name__ == "__main__":
    main()
