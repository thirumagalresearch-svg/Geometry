# Project Status: Outputs and Development Log

This document details the expected outputs of the **Geometry-Aware Selfie Perspective Rectification** project and provides a comprehensive log of the setup, debugging, development, and cleanup actions completed.

---

## 📸 Expected Project Outputs

When you run this project, it produces multiple research and visual outputs:

### 1. Visual Outputs (Selfie Rectification)
For every input selfie processed, the pipeline generates:
* **`landmarks.png`**: The original image with localized 2D face mesh keypoints overlaid.
* **`depth.png`**: A colorized relative depth map representing the 3D structure of the face (via Depth Anything V2).
* **`rectified.png`**: The perspective-corrected face where wide-angle facial distortion (e.g., enlarged noses, narrow faces) is warped back into natural facial proportions using a camera projection matrix.
* **`codeformer/` output**: High-quality face restoration results that reconstruct facial details (eyes, nose, skin textures) potentially blurred or distorted during geometric warping.
* **`final.png`**: A photorealistic image generated via Stable Diffusion v1.5 img2img, blending the geometric rectification with realistic textures at a controlled denoising strength (0.2).

### 2. User Interfaces & CLI
* **Web UI (Gradio)**: Launches an interactive local webpage via `.\launch.bat` (at `http://127.0.0.1:7860`) where you can:
  * Drag-and-drop selfie images.
  * Click a single button to run the entire pipeline.
  * View side-by-side comparisons of the original vs final rectified images.
  * Inspect the step-by-step intermediate visualizations of the pipeline stages.
  * Download the final rectified image.
* **CLI Tool (`main.py`)**: A command-line script to run the rectification pipeline programmatically in bulk:
  `python -m scripts.main <input_path>`

### 3. Research Assets & Models
* **RK4 Integrator (`models/rk4_optimizer.py`)**: A starter Runge-Kutta 4th-order integration and trajectory optimization module ready for differentiable training of dynamic face/pose deformation fields.
* **3D Geometry (`scripts/geometry.py`)**: Utilities to project 3D points back to 2D coordinates and calculate camera matrices.
* **3D Reconstruction placeholders**: Pre-configured placeholders for lightweight Gaussian Splatting and NeRF models optimized for VRAM constraints.

---

## 🛠️ Complete Development Log

Here is the log of all fixes, integrations, and optimizations implemented since yesterday:

### 1. Environment & GPU Verification
* Diagnosed and verified the Python virtual environment (`.venv`).
* Validated GPU and CUDA integration (**NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM)** running **CUDA 11.8** and **PyTorch 2.2.2+cu118**).
* Fixed python terminal pathing issues so that modules can be run cleanly as `python -m scripts.<script_name>`.

### 2. Unicode Error Fixes on Windows CMD
* Detected `UnicodeEncodeError` crashes occurring under the default Windows command prompt encoding (CP1252/cp1252) when printing emojis (e.g., `✓`, `⚠`, `📸`, `✅`).
* Replaced all unicode emojis and symbols in `verify_setup.py`, `main.py`, and `pipeline.py` with ASCII logging prefixes (e.g., `[OK]`, `[WARN]`, `[ERROR]`, `[INFO]`) to ensure the scripts run robustly in standard CMD shells.

### 3. CodeFormer Wrapper & Dependency Debugging
* **BasicSR version.py Generation**: Resolved `ModuleNotFoundError: No module named 'basicsr.version'` by programmatically generating the missing `version.py` file inside `repos/CodeFormer/basicsr/` without requiring heavy CUDA compiling.
* **Absolute Path Mapping**: Resolved path-finding issues by modifying `codeformer_wrapper.py` to resolve absolute paths (`.resolve()`) for input and output parameters. This ensures that CodeFormer can locate images when subprocess CWD is set to the repository folder.
* **Output Folder Verification**: Corrected the output path search pattern in `codeformer_wrapper.py` to look inside `final_results/<image_name>.png`, matching CodeFormer's actual file saving convention.

### 4. Cache Warming
* Warm-cached the pre-trained weights for **Depth Anything V2 Small** and **Stable Diffusion v1.5** (FP16 mode) locally to avoid runtime download delays.
* Set up **InsightFace** buffalo_l model weights in the user home directory (`~/.insightface/`).

### 5. Automated Unit Tests Verification
Wrote and successfully verified tests for every module:
* `test_gpu.py` (GPU & CUDA verification) — **PASSED**
* `test_mediapipe.py` (Face mesh landmarks) — **PASSED**
* `test_depth.py` (Relative depth mapping) — **PASSED**
* `test_codeformer.py` (Face quality restoration) — **PASSED**
* `test_diffusion.py` (Stable Diffusion image refinement) — **PASSED**

### 6. Interactive UI Enhancement
* Rewrote [gradio_app.py](file:///e:/Hackathon2025-2028/selfi-destro/UnDistort-Selfie/SelfieRectification/scripts/gradio_app.py) to provide a premium user experience:
  * Sleek custom dark mode stylesheet utilizing radial gradients and custom fonts.
  * Side-by-side inputs and rectified results.
  * 4-column step-by-step visualization layout showcasing the landmarks, depth map, perspective warp, and CodeFormer stages.
  * Standard file download container for easy downloading.
  * Technical overview section detailing the RTX 3050 VRAM optimizations (FP16, CPU offload, attention slicing).

### 7. Hugging Face Dataset Downloader Integration
* Installed the Hugging Face `datasets` library.
* Created [download_datasets.py](file:///e:/Hackathon2025-2028/selfi-destro/UnDistort-Selfie/SelfieRectification/scripts/download_datasets.py) to stream and cache sample images from the academic benchmark datasets mentioned in the paper:
  * **CelebA-HQ** (`mattymchen/celeba-hq` dataset) saved to `datasets/celeba_hq/`
  * **FFHQ-256** (`merkol/ffhq-256` dataset) saved to `datasets/ffhq/`
* Successfully tested and validated pipeline runs on these downloaded benchmark dataset images.

### 8. Project Directory Cleanup
* Deleted redundant and extra markdown documentation/logs (`BUILD_SUMMARY.md`, `ERROR_HANDLING.md`, `QUALITY_REPORT.md`, `QUICKSTART.txt`) from the project root to keep the workspace clean, lightweight, and professional.
