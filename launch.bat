@echo off
setlocal
set ROOT=%~dp0
set HF_HOME=%ROOT%cache\huggingface
set TRANSFORMERS_CACHE=%HF_HOME%\transformers
set DIFFUSERS_CACHE=%HF_HOME%\diffusers
set HF_HUB_ENABLE_HF_TRANSFER=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

call "%ROOT%.venv\Scripts\activate"
python -m scripts.gradio_app
endlocal
