import os
import urllib.request
from pathlib import Path
import zipfile
import shutil

# Import config paths
from scripts.config import CHECKPOINTS_DIR, REPOS_DIR, setup_env, ensure_dirs

def download_file(url: str, dest_path: Path) -> None:
    """Download a file with progress updates."""
    if dest_path.exists():
        print(f"Already exists: {dest_path.name}")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} to {dest_path}...")
    
    def progress_callback(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = min(100.0, read_so_far * 100 / total_size)
            print(f"\rProgress: {percent:.1f}% ({read_so_far/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB)", end="")
        else:
            print(f"\rDownloaded {read_so_far/(1024*1024):.1f}MB", end="")
            
    try:
        urllib.request.urlretrieve(url, str(dest_path), progress_callback)
        print("\nDownload completed successfully.")
    except Exception as e:
        if dest_path.exists():
            dest_path.unlink() # Delete partial file on failure
        raise RuntimeError(f"Failed to download {url}: {str(e)}") from e

def setup_codeformer_weights() -> None:
    """Download and place CodeFormer weights in the checkpoints folder and CodeFormer repository."""
    print("\n--- Setting up CodeFormer weights ---")
    urls = {
        "codeformer.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "detection_Resnet50_Final.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "parsing_parsenet.pth": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth"
    }

    cf_checkpoints = CHECKPOINTS_DIR / "CodeFormer"
    cf_checkpoints.mkdir(parents=True, exist_ok=True)
    
    download_file(urls["codeformer.pth"], cf_checkpoints / "codeformer.pth")
    download_file(urls["detection_Resnet50_Final.pth"], cf_checkpoints / "detection_Resnet50_Final.pth")
    download_file(urls["parsing_parsenet.pth"], cf_checkpoints / "parsing_parsenet.pth")

    # Copy to the cloned CodeFormer repo directory if it exists
    cf_repo = REPOS_DIR / "CodeFormer"
    if cf_repo.exists():
        print("Copying weights to CodeFormer repository...")
        cf_weights_dir = cf_repo / "weights" / "CodeFormer"
        facelib_weights_dir = cf_repo / "weights" / "facelib"
        
        cf_weights_dir.mkdir(parents=True, exist_ok=True)
        facelib_weights_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(cf_checkpoints / "codeformer.pth", cf_weights_dir / "codeformer.pth")
        shutil.copy2(cf_checkpoints / "detection_Resnet50_Final.pth", facelib_weights_dir / "detection_Resnet50_Final.pth")
        shutil.copy2(cf_checkpoints / "parsing_parsenet.pth", facelib_weights_dir / "parsing_parsenet.pth")
        print("CodeFormer repository weights updated.")
    else:
        print("Warning: CodeFormer repo not found in repos/, skipped copying weights.")

def setup_insightface_weights() -> None:
    """Download and place InsightFace models in user home directory."""
    print("\n--- Setting up InsightFace weights ---")
    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    
    # Target path: ~/.insightface/models/buffalo_l
    user_home = Path.home()
    insightface_dir = user_home / ".insightface" / "models" / "buffalo_l"
    
    if insightface_dir.exists() and any(insightface_dir.glob("*.onnx")):
        print("InsightFace buffalo_l models already exist.")
        return
        
    zip_dest = CHECKPOINTS_DIR / "buffalo_l.zip"
    download_file(url, zip_dest)
    
    print("Extracting InsightFace model zip...")
    insightface_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_dest, 'r') as zip_ref:
        zip_ref.extractall(insightface_dir)
        
    # Delete temporary zip
    if zip_dest.exists():
        zip_dest.unlink()
    print("InsightFace models extracted to user home.")

def warm_huggingface_cache() -> None:
    """Warm HF cache with Depth Anything V2 Small and Stable Diffusion v1.5."""
    print("\n--- Warming Hugging Face Caches (Depth Anything V2 & Stable Diffusion v1.5) ---")
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        print("Loading Depth Anything V2 Small model and processor...")
        AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        print("Depth Anything V2 Small cached successfully.")
    except Exception as e:
        print(f"Warning: Failed to warm Depth Anything cache: {str(e)}")

    try:
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline
        from scripts.config import DTYPE
        print("Loading Stable Diffusion v1.5 in FP16 (might take a bit to download/cache)...")
        StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=DTYPE,
            safety_checker=None
        )
        print("Stable Diffusion v1.5 cached successfully.")
    except Exception as e:
        print(f"Warning: Failed to warm Stable Diffusion cache: {str(e)}")

def main() -> None:
    setup_env()
    ensure_dirs()
    setup_codeformer_weights()
    setup_insightface_weights()
    warm_huggingface_cache()
    print("\nAll pretrained weights set up successfully!")

if __name__ == "__main__":
    main()
