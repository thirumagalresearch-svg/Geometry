import os
import shutil
import subprocess
from pathlib import Path

# Import config paths
from scripts.config import REPOS_DIR, ensure_dirs, setup_env

REPOS = {
    "mediapipe": "https://github.com/google-ai-edge/mediapipe",
    "Depth-Anything-V2": "https://github.com/DepthAnything/Depth-Anything-V2",
    "NextFace": "https://github.com/abdallahdib/NextFace",
    "gaussian-splatting": "https://github.com/graphdeco-inria/gaussian-splatting",
    "CodeFormer": "https://github.com/sczhou/CodeFormer",
    "diffusers": "https://github.com/huggingface/diffusers",
    "insightface": "https://github.com/deepinsight/insightface"
}

def clone_repo(name: str, url: str) -> None:
    dest = REPOS_DIR / name
    print(f"\n=== Processing {name} ===")
    
    # If the directory exists but is empty, delete it so git clone can succeed
    if dest.exists():
        is_empty = True
        for root, dirs, files in os.walk(dest):
            if files:
                is_empty = False
                break
        if is_empty:
            print(f"Directory {dest} is empty. Removing to clone fresh...")
            shutil.rmtree(dest)
        else:
            print(f"Repository {name} already exists and contains files.")
            return

    print(f"Cloning {url} into {dest}...")
    try:
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
            env=env
        )
        print(f"Successfully cloned {name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone {name}: {str(e)}")

def main() -> None:
    setup_env()
    ensure_dirs()
    
    try:
        subprocess.run(["git", "config", "--global", "core.longpaths", "true"], check=True)
        print("Enabled Git core.longpaths globally.")
    except Exception as e:
        print(f"Warning: Could not set git config core.longpaths: {e}")

    for name, url in REPOS.items():
        clone_repo(name, url)

if __name__ == "__main__":
    main()
