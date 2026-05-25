# Installation & Setup Guide

This guide explains how to install and run the SelfieRectification pipeline on Windows and macOS.

## System Requirements

### Windows
- Windows 10 or 11
- Python 3.10
- NVIDIA GPU with CUDA 11.8+ drivers
- Git installed

### macOS
- macOS 12 or later
- Python 3.10 or 3.11
- Apple Silicon or Intel CPU
- PyTorch MPS/CPU support for Mac

---

## Recommended Project Structure

```
SelfieRectification/
├── .venv/                  # Python virtual environment
├── cache/                  # HuggingFace cache
├── checkpoints/            # Model weights and checkpoints
├── datasets/               # Input and sample images
├── outputs/                # Generated results
├── repos/                  # Cloned repositories
├── requirements.txt        # Windows/CUDA dependencies
├── requirements-mac.txt    # macOS/CPU dependencies
├── launch.bat              # Windows Gradio launcher
├── README.md               # User-facing project docs
└── INSTALL.md              # This installation guide
```

---

## Windows Installation

```powershell
cd SelfieRectification
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If CUDA installation fails, install the PyTorch CUDA wheel manually:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt --no-deps
```

---

## macOS Installation

```bash
cd SelfieRectification
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio
python3 -m pip install -r requirements-mac.txt
```

> If you are using Apple Silicon, install the correct PyTorch wheel from https://pytorch.org before installing the rest of the dependencies.

---

## Verify Installation

After dependencies are installed, run:

```bash
python -m scripts.verify_setup
python -m scripts.test_gpu
```

Expected checks:
- Python and virtual environment are active
- Required Python packages are installed
- PyTorch and GPU support are available when applicable
- Hugging Face cache paths are valid

---

## Running the Pipeline

### Web UI

#### Windows
```powershell
cd SelfieRectification
.\.venv\Scripts\Activate.ps1
.\launch.bat
```
Open `http://localhost:7860`

#### macOS
```bash
cd SelfieRectification
source .venv/bin/activate
python -m scripts.gradio_app
```

### Command-line

```bash
cd SelfieRectification
source .venv/bin/activate    # or .\.venv\Scripts\Activate.ps1 on Windows
python -m scripts.main datasets/input_sample.jpg
```

---

## Sample Output Files

The pipeline saves processed images to `outputs/<timestamp>/`. Common output files include:
- `landmarks.png`
- `depth.png`
- `rectified.png`
- `final.png`
- `outputs/reports/*.html`

Sample input and output files are included in `datasets/` for quick verification.

---

## Common Issues and Fixes

### CUDA not detected on Windows
- Update NVIDIA drivers.
- Confirm CUDA 11.8 is installed.
- Verify `nvidia-smi` works.

### Missing module errors
```bash
python -m pip install <package> --upgrade --force-reinstall
```

### Out of memory (OOM)
- Lower resolution in `scripts/config.py`
- Reduce diffusion steps in `scripts/diffusion_refine.py`
- Run only the geometric correction stages if GPU memory is limited

### xFormers installation fails
- xFormers is optional.
- The pipeline can fall back to attention slicing.
- Install with `python -m pip install xformers --no-deps` if needed.

---

## GitHub push workflow

### First-time push
```bash
cd SelfieRectification
git init
git add .
git commit -m "Add setup instructions for Windows and macOS"
git branch -M main
git remote add origin https://github.com/<username>/<repo>.git
git push -u origin main
```

### Subsequent updates
```bash
git add README.md INSTALL.md requirements-mac.txt
git commit -m "Update installation docs and cleanup instructions"
git push
```

---

## Notes
- Keep generated directories out of Git: `.venv/`, `cache/`, `checkpoints/`, `outputs/`, `repos/`.
- Use `requirements.txt` for Windows/CUDA installs.
- Use `requirements-mac.txt` for macOS CPU/MPS installs.
- The Gradio UI and CLI share the same core pipeline.

## Support

If issues persist:

1. Check `scripts/verify_setup.py` output
2. Review GPU memory: `nvidia-smi`
3. Check error logs in `outputs/`
4. Verify all repos cloned in `repos/`
