import os
import sys
from pathlib import Path

# Remove local 'datasets' directory from sys.path to avoid name shadowing
sys.path = [p for p in sys.path if p not in ('', '.')]

from datasets import load_dataset
from PIL import Image

def main():
    root = Path(__file__).resolve().parents[1]
    datasets_dir = root / "datasets"
    
    # 1. CelebA-HQ
    celeba_dir = datasets_dir / "celeba_hq"
    celeba_dir.mkdir(parents=True, exist_ok=True)
    print("Streaming CelebA-HQ images from Hugging Face...")
    try:
        ds_celeba = load_dataset("mattymchen/celeba-hq", split="train", streaming=True)
        count = 0
        for i, item in enumerate(ds_celeba):
            if count >= 5:
                break
            img = item["image"]
            img_path = celeba_dir / f"celeba_hq_{i}.jpg"
            img.convert("RGB").save(img_path)
            print(f"Saved: {img_path}")
            count += 1
    except Exception as e:
        print(f"Error downloading CelebA-HQ: {e}")

    # 2. FFHQ
    ffhq_dir = datasets_dir / "ffhq"
    ffhq_dir.mkdir(parents=True, exist_ok=True)
    print("Streaming FFHQ images from Hugging Face...")
    try:
        ds_ffhq = load_dataset("merkol/ffhq-256", split="train", streaming=True)
        count = 0
        for i, item in enumerate(ds_ffhq):
            if count >= 5:
                break
            img = item["image"]
            img_path = ffhq_dir / f"ffhq_{i}.png"
            img.convert("RGB").save(img_path)
            print(f"Saved: {img_path}")
            count += 1
    except Exception as e:
        print(f"Error downloading FFHQ: {e}")

    print("Datasets sample download completed!")

if __name__ == "__main__":
    main()
