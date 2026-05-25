from typing import Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from scripts.config import DEVICE, DTYPE


class DepthAnythingV2:
    """Lightweight Depth Anything V2 (small) wrapper.
    
    Raises:
        RuntimeError: If model loading fails
    """

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf") -> None:
        """Initialize depth model.
        
        Args:
            model_id: HuggingFace model identifier
        
        Raises:
            RuntimeError: If model cannot be loaded
        """
        self.model_id = model_id
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(
                model_id,
                torch_dtype=DTYPE,
            )
            self.model.to(DEVICE)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load Depth Anything V2 model: {str(e)}") from e

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Tuple[np.ndarray, Image.Image]:
        """Return normalized depth array and visualization image.
        
        Args:
            image: PIL Image in RGB format
        
        Returns:
            Tuple of (depth_array, depth_visualization)
        
        Raises:
            TypeError: If image is not PIL Image
            RuntimeError: If prediction fails
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image)}")
        
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            autocast = torch.autocast(
                device_type="cuda",
                dtype=DTYPE,
                enabled=(DEVICE == "cuda")
            )
            with autocast:
                outputs = self.model(**inputs)

            predicted = outputs.predicted_depth
            predicted = torch.nn.functional.interpolate(
                predicted.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)

            depth = predicted.squeeze(0).cpu().numpy()
            depth_min = depth.min()
            depth_max = depth.max()
            
            if depth_max <= depth_min:
                raise ValueError(f"Invalid depth range: min={depth_min}, max={depth_max}")
            
            depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
            depth_vis = (depth_norm * 255).astype(np.uint8)
            return depth_norm, Image.fromarray(depth_vis)
        
        except Exception as e:
            raise RuntimeError(f"Depth estimation failed: {str(e)}") from e
