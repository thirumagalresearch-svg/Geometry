"""Performance benchmarking and profiling utilities."""
import time
from pathlib import Path
from typing import Callable, Dict

import torch

from scripts.config import DEVICE


def benchmark_model(
    model_fn: Callable,
    input_data: torch.Tensor,
    num_runs: int = 5,
    warmup: int = 2,
) -> Dict[str, float]:
    """Benchmark a model's inference time and memory usage."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model_fn(input_data)

    torch.cuda.synchronize() if DEVICE == "cuda" else None

    times = []
    for _ in range(num_runs):
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize() if DEVICE == "cuda" else None

        start = time.perf_counter()
        with torch.no_grad():
            _ = model_fn(input_data)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        end = time.perf_counter()

        times.append(end - start)

    return {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated() / 1e6
            if DEVICE == "cuda"
            else 0
        ),
    }


def profile_pipeline(input_image_path: Path) -> None:
    """Profile each stage of the pipeline."""
    from PIL import Image

    from scripts.depth_anything import DepthAnythingV2
    from scripts.mediapipe_landmarks import extract_landmarks
    from scripts.utils import load_image

    print("Pipeline Profiling")
    print("=" * 60)

    image = load_image(input_image_path)

    # MediaPipe landmarks (CPU)
    start = time.perf_counter()
    landmarks = extract_landmarks(image)
    elapsed = time.perf_counter() - start
    print(f"MediaPipe Landmarks: {elapsed:.3f}s")

    # Depth estimation
    depth_model = DepthAnythingV2()
    start = time.perf_counter()
    _, _ = depth_model.predict(image)
    elapsed = time.perf_counter() - start
    print(f"Depth Anything V2:   {elapsed:.3f}s")

    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to test image")
    args = parser.parse_args()
    profile_pipeline(Path(args.image))
