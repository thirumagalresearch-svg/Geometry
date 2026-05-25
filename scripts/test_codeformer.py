import argparse
from pathlib import Path

from scripts.config import setup_env
from scripts.codeformer_wrapper import run_codeformer


def main() -> None:
    """Run CodeFormer on a single image."""
    setup_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input selfie image.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir = image_path.parent / "codeformer_test"
    output_path = run_codeformer(image_path, output_dir, fidelity_weight=0.5)
    print(f"CodeFormer output: {output_path}")


if __name__ == "__main__":
    main()
