"""Installation and setup verification utilities."""
import subprocess
import sys
from pathlib import Path

from scripts.config import DEVICE, ensure_dirs, setup_env


def check_torch() -> None:
    """Verify PyTorch and CUDA setup."""
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {DEVICE}\n")


def check_models() -> None:
    """Verify required models can be imported."""
    try:
        import mediapipe as mp

        print("[OK] MediaPipe installed")
    except ImportError:
        print("[FAIL] MediaPipe missing (run: pip install mediapipe)")

    try:
        import cv2

        print("[OK] OpenCV installed")
    except ImportError:
        print("[FAIL] OpenCV missing (run: pip install opencv-python)")

    try:
        from transformers import AutoModelForDepthEstimation

        print("[OK] Transformers installed")
    except ImportError:
        print("[FAIL] Transformers missing (run: pip install transformers)")

    try:
        from diffusers import StableDiffusionImg2ImgPipeline

        print("[OK] Diffusers installed")
    except ImportError:
        print("[FAIL] Diffusers missing (run: pip install diffusers)")

    try:
        import gradio as gr

        print("[OK] Gradio installed\n")
    except ImportError:
        print("[FAIL] Gradio missing (run: pip install gradio)\n")


def verify_repos() -> None:
    """Check that cloned repositories exist."""
    from scripts.config import REPOS_DIR

    repos = [
        "mediapipe",
        "Depth-Anything-V2",
        "CodeFormer",
        "gaussian-splatting",
        "diffusers",
        "insightface",
    ]
    print("Repository status:")
    for repo in repos:
        path = REPOS_DIR / repo
        status = "[OK]" if path.exists() else "[MISSING]"
        print(f"  {status} {repo}")
    print()


def verify_setup() -> None:
    """Run all verification checks."""
    setup_env()
    ensure_dirs()

    print("=" * 60)
    print("SETUP VERIFICATION")
    print("=" * 60)
    print()

    check_torch()
    check_models()
    verify_repos()

    print("=" * 60)
    print("Next steps:")
    print("  1. pip install -r requirements.txt")
    print("  2. python -m scripts.test_gpu")
    print("  3. python -m scripts.main datasets/sample.jpg")
    print("  4. ./launch.bat  (for Gradio UI)")
    print("=" * 60)


if __name__ == "__main__":
    verify_setup()
