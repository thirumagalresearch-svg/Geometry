import argparse
from pathlib import Path

from scripts.config import setup_env
from scripts.diffusion_refine import DiffusionRefiner
from scripts.utils import load_image, save_image


def main() -> None:
    """Run a lightweight SD img2img refinement test."""
    setup_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input selfie image.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    refiner = DiffusionRefiner()
    output = refiner.refine(load_image(image_path))

    output_path = image_path.with_name("diffusion_test.png")
    save_image(output, output_path)
    print(f"Diffusion output: {output_path}")


if __name__ == "__main__":
    main()
