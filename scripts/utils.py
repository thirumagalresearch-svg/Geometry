from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> Image.Image:
    """Load an image as RGB."""
    return Image.open(path).convert("RGB")


def save_image(image: Image.Image, path: Path) -> None:
    """Save an RGB image to disk."""
    ensure_dir(path.parent)
    image.save(path)


def to_numpy_rgb(image: Image.Image) -> np.ndarray:
    """Convert PIL image to RGB numpy array."""
    return np.array(image, dtype=np.uint8)


def from_numpy_rgb(array: np.ndarray) -> Image.Image:
    """Convert RGB numpy array to PIL image."""
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def scale_to_square(image: Image.Image, size: int) -> Image.Image:
    """Resize image to a square while preserving center content."""
    return image.resize((size, size), Image.LANCZOS)
