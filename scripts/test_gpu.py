import torch

from scripts.config import DEVICE, setup_env


def main() -> None:
    """Print GPU and CUDA status."""
    setup_env()
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    print(f"Selected device: {DEVICE}")


if __name__ == "__main__":
    main()
