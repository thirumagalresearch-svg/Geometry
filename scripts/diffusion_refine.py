from typing import Optional

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

from scripts.config import DEVICE, DTYPE, LOW_VRAM, SD_RESOLUTION
from scripts.utils import scale_to_square


class DiffusionRefiner:
    """Lightweight SD v1.5 img2img refiner for 6GB VRAM.
    
    Raises:
        RuntimeError: If pipeline initialization fails
    """

    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5") -> None:
        """Initialize diffusion pipeline.
        
        Args:
            model_id: HuggingFace model identifier
        
        Raises:
            RuntimeError: If model cannot be loaded
        """
        self.model_id = model_id
        try:
            self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=DTYPE,
                safety_checker=None,
            )
            self.pipeline.set_progress_bar_config(disable=True)
            self._configure_pipeline()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize diffusion pipeline: {str(e)}") from e

    def _configure_pipeline(self) -> None:
        """Apply low VRAM and memory-efficient settings."""
        try:
            if DEVICE == "cuda":
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_vae_slicing()
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                except (ImportError, RuntimeError):
                    # xFormers is optional; slicing is still enabled.
                    pass
                if LOW_VRAM:
                    self.pipeline.enable_model_cpu_offload()
                else:
                    self.pipeline.to(DEVICE)
            else:
                self.pipeline.to("cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to configure diffusion pipeline: {str(e)}") from e

    def refine(
        self,
        image: Image.Image,
        prompt: str = "a realistic selfie photo, natural skin texture, soft lighting",
        negative_prompt: str = "low quality, blurry, deformed, artifacts",
        strength: float = 0.35,
        guidance_scale: float = 3.0,
        steps: int = 20,
    ) -> Image.Image:
        """Run img2img refinement and return the output image.
        
        Args:
            image: Input PIL Image
            prompt: Positive prompt for diffusion
            negative_prompt: Negative prompt for diffusion
            strength: Denoising strength (0-1)
            guidance_scale: Classifier-free guidance scale
            steps: Number of diffusion steps
        
        Returns:
            Refined PIL Image
        
        Raises:
            TypeError: If image not PIL Image
            ValueError: If parameters invalid
            RuntimeError: If inference fails
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image)}")
        
        if not (0 <= strength <= 1):
            raise ValueError(f"strength must be in [0, 1], got {strength}")
        
        if guidance_scale < 0:
            raise ValueError(f"guidance_scale must be >= 0, got {guidance_scale}")
        
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        
        try:
            image = scale_to_square(image, SD_RESOLUTION)
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
            )
            return result.images[0]
        except Exception as e:
            raise RuntimeError(f"Diffusion refinement failed: {str(e)}") from e
