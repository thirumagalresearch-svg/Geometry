import argparse
from pathlib import Path

from scripts.config import setup_env
from scripts.depth_anything import DepthAnythingV2
from scripts.utils import load_image, save_image


def main() -> None:
    """Run a lightweight depth estimation test."""
    setup_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input selfie image.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = DepthAnythingV2()
    _, depth_vis = model.predict(load_image(image_path))

    output_path = image_path.with_name("depth_test.png")
    save_image(depth_vis.convert("RGB"), output_path)
    print(f"Depth map saved to: {output_path}")


if __name__ == "__main__":
    main()
