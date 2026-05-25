import os
from pathlib import Path

import torch

# Project root directory (SelfieRectification).
ROOT = Path(__file__).resolve().parents[1]

# Core folders.
DATASETS_DIR = ROOT / "datasets"
CHECKPOINTS_DIR = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"
REPOS_DIR = ROOT / "repos"
CACHE_DIR = ROOT / "cache"

# Hugging Face cache location.
HF_HOME = CACHE_DIR / "huggingface"

# Runtime settings.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
LOW_VRAM = True
SD_RESOLUTION = 512


def setup_env() -> None:
    """Configure environment variables for caching and CUDA behavior."""
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "transformers"))
    os.environ.setdefault("DIFFUSERS_CACHE", str(HF_HOME / "diffusers"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")


def ensure_dirs() -> None:
    """Ensure required directories exist."""
    for path in [
        DATASETS_DIR,
        CHECKPOINTS_DIR,
        OUTPUTS_DIR,
        MODELS_DIR,
        REPOS_DIR,
        CACHE_DIR,
        HF_HOME,
    ]:
        path.mkdir(parents=True, exist_ok=True)
