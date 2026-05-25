import subprocess
import sys
from pathlib import Path
from typing import Optional

from scripts.config import REPOS_DIR


def _codeformer_repo() -> Path:
    """Return CodeFormer repository path.
    
    Raises:
        RuntimeError: If repository doesn't exist
    """
    repo = REPOS_DIR / "CodeFormer"
    if not repo.exists():
        raise RuntimeError(f"CodeFormer repository not found at {repo}")
    return repo


def ensure_codeformer_weights() -> None:
    """Download CodeFormer and facelib weights using upstream script.
    
    Raises:
        RuntimeError: If weights cannot be downloaded
    """
    repo = _codeformer_repo()
    script = repo / "scripts" / "download_pretrained_models.py"
    if not script.exists():
        raise FileNotFoundError(f"CodeFormer script not found: {script}")

    weights_dir = repo / "weights" / "CodeFormer"
    if not any(weights_dir.glob("*.pth")):
        try:
            subprocess.run(
                [sys.executable, str(script), "CodeFormer"],
                check=True,
                cwd=repo,
                capture_output=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("CodeFormer weight download timed out (>5 minutes)") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"CodeFormer weight download failed: {e.stderr.decode() if e.stderr else str(e)}") from e

    facelib_dir = repo / "weights" / "facelib"
    if not any(facelib_dir.glob("*.pth")):
        try:
            subprocess.run(
                [sys.executable, str(script), "facelib"],
                check=True,
                cwd=repo,
                capture_output=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError("facelib weight download timed out (>5 minutes)") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"facelib weight download failed: {e.stderr.decode() if e.stderr else str(e)}") from e


def run_codeformer(
    input_path: Path,
    output_dir: Path,
    fidelity_weight: float = 0.95,
    face_upsample: bool = False,
) -> Path:
    """Run CodeFormer inference on a single image and return output path.
    
    Args:
        input_path: Path to input image
        output_dir: Directory to save output
        fidelity_weight: Fidelity vs quality balance (0-1)
        face_upsample: Whether to upsample face
    
    Returns:
        Path to output image
    
    Raises:
        FileNotFoundError: If input or script not found
        ValueError: If fidelity_weight invalid
        RuntimeError: If CodeFormer inference fails
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    
    if not (0 <= fidelity_weight <= 1):
        raise ValueError(f"fidelity_weight must be in [0, 1], got {fidelity_weight}")

    repo = _codeformer_repo()
    script = repo / "inference_codeformer.py"
    if not script.exists():
        raise FileNotFoundError(f"CodeFormer inference script missing: {script}")

    ensure_codeformer_weights()

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        "--input_path",
        str(input_path.resolve()),
        "--output_path",
        str(output_dir.resolve()),
        "-w",
        str(fidelity_weight),
    ]
    if face_upsample:
        cmd.append("--face_upsample")

    try:
        subprocess.run(cmd, check=True, cwd=repo, capture_output=True, timeout=600)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("CodeFormer inference timed out (>10 minutes)") from e
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else "Unknown error"
        raise RuntimeError(f"CodeFormer inference failed: {stderr}") from e

    output_path = output_dir / "final_results" / f"{input_path.stem}.png"
    if not output_path.exists():
        # Fallback to check directly in output_dir
        direct_path = output_dir / f"{input_path.stem}.png"
        if direct_path.exists():
            return direct_path
        raise RuntimeError(f"CodeFormer did not produce output at {output_path}")
    return output_path
