import cv2
import numpy as np
from PIL import Image
from typing import Optional, Dict, Tuple

# Landmark indices for facial features in MediaPipe Face Mesh (468 landmarks)
LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW_INDICES = [70, 63, 105, 66, 107, 55, 117, 124, 46, 53]
RIGHT_EYEBROW_INDICES = [300, 293, 334, 296, 336, 285, 346, 353, 276, 283]
LIPS_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 95, 88]

# Jawline landmarks + lower mouth landmarks for beard/lower face masking
BEARD_INDICES = [
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 
    377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 
    324, 326, 2, 97, 162, 127
]


def make_polygon_mask(shape: Tuple[int, int], points: np.ndarray, dilate_pixels: int = 0, blur_pixels: int = 0) -> np.ndarray:
    """Helper to create a smoothed binary mask from a set of 2D points."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if len(points) >= 3:
        pts = points[:, :2].astype(np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
        
        if dilate_pixels > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_pixels, dilate_pixels))
            mask = cv2.dilate(mask, kernel)
            
        if blur_pixels > 0:
            ksize = blur_pixels | 1
            mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (ksize, ksize), 0)
        else:
            mask = mask.astype(np.float32) / 255.0
    else:
        mask = mask.astype(np.float32)
    return mask


def generate_feature_masks(image_shape: Tuple[int, int], landmarks_all: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """Generate localized facial feature masks using 468 mesh landmarks.
    
    If landmarks_all is None, returns empty/zero masks.
    """
    h, w = image_shape[:2]
    masks = {
        "left_eye": np.zeros((h, w), dtype=np.float32),
        "right_eye": np.zeros((h, w), dtype=np.float32),
        "left_eyebrow": np.zeros((h, w), dtype=np.float32),
        "right_eyebrow": np.zeros((h, w), dtype=np.float32),
        "lips": np.zeros((h, w), dtype=np.float32),
        "beard": np.zeros((h, w), dtype=np.float32),
        "face": np.zeros((h, w), dtype=np.float32),
    }
    
    if landmarks_all is None or len(landmarks_all) < 468:
        return masks
        
    try:
        # 1. Left/Right Eyes (with slight dilation to include eyelashes/eyelids)
        masks["left_eye"] = make_polygon_mask(image_shape, landmarks_all[LEFT_EYE_INDICES], dilate_pixels=7, blur_pixels=5)
        masks["right_eye"] = make_polygon_mask(image_shape, landmarks_all[RIGHT_EYE_INDICES], dilate_pixels=7, blur_pixels=5)
        
        # 2. Eyebrows
        masks["left_eyebrow"] = make_polygon_mask(image_shape, landmarks_all[LEFT_EYEBROW_INDICES], dilate_pixels=5, blur_pixels=5)
        masks["right_eyebrow"] = make_polygon_mask(image_shape, landmarks_all[RIGHT_EYEBROW_INDICES], dilate_pixels=5, blur_pixels=5)
        
        # 3. Lips
        masks["lips"] = make_polygon_mask(image_shape, landmarks_all[LIPS_INDICES], dilate_pixels=5, blur_pixels=5)
        
        # 4. Beard/Jaw Area
        masks["beard"] = make_polygon_mask(image_shape, landmarks_all[BEARD_INDICES], dilate_pixels=15, blur_pixels=11)
        
        # 5. Whole face mesh convex hull
        masks["face"] = make_polygon_mask(image_shape, landmarks_all, dilate_pixels=0, blur_pixels=15)
        
    except Exception as e:
        print(f"[WARN] Failed to generate feature masks from landmarks: {e}")
        
    return masks


def extract_high_frequency_details(img_np: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Extract high-frequency detail residual from an RGB image."""
    img_float = img_np.astype(np.float32)
    # Extract details by subtracting Gaussian blur
    blurred = cv2.GaussianBlur(img_float, (ksize, ksize), sigma)
    residual = img_float - blurred
    return residual


def enhance_details(
    original_rectified: Image.Image,
    refined: Image.Image,
    landmarks_all: Optional[np.ndarray],
    blend_weight: float = 0.6,
    sharpen_strength: float = 0.3,
    eye_boost: float = 0.4,
    hair_preservation: float = 0.5,
) -> Image.Image:
    """Apply high-frequency detail blending and edge-aware facial feature sharpening.
    
    Args:
        original_rectified: High-quality pre-diffusion image (CodeFormer restored or perspective rectified)
        refined: Low-noise, slightly smoothed post-diffusion image
        landmarks_all: 468 face mesh landmarks from perspective-rectified image
        blend_weight: Scale factor for high-frequency details (0 to 1)
        sharpen_strength: Scale factor for edge-aware sharpening (0 to 1)
        eye_boost: Extra sharpening boost for eye region (0 to 1)
        hair_preservation: Blend strength for hair high-frequency details (0 to 1)
        
    Returns:
        Detail-enhanced, sharp PIL Image
    """
    rect_np = np.array(original_rectified.convert("RGB"), dtype=np.uint8)
    ref_np = np.array(refined.convert("RGB"), dtype=np.uint8)
    h, w = rect_np.shape[:2]
    
    # 1. Extract High-Frequency Details from pre-diffusion image
    details = extract_high_frequency_details(rect_np, ksize=5, sigma=1.0)
    
    # Calculate texture-adaptive mask based on local variance of high-frequency details
    detail_magnitude = np.mean(np.abs(details), axis=-1)
    texture_mask = cv2.GaussianBlur(detail_magnitude, (9, 9), 3.0)
    texture_mask = np.clip(texture_mask / 6.0, 0.0, 1.0) # normalize to typical texture intensity
    
    # Generate spatial feature masks
    masks = generate_feature_masks((h, w), landmarks_all)
    face_mask = masks["face"]
    
    # 2. Blend Details Back (Texture-Adaptive Blending)
    # We blend details 100% on the face region, and selectively in the hair/background using the texture mask.
    # This prevents background noise amplification while fully keeping hair strands and facial details.
    blend_map = face_mask + (1.0 - face_mask) * texture_mask * hair_preservation
    blend_map = np.clip(blend_map, 0.0, 1.0)
    
    ref_float = ref_np.astype(np.float32)
    blended = ref_float + blend_weight * details * blend_map[:, :, np.newaxis]
    blended_np = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    
    # 3. Localized Edge-Aware Sharpening
    # Detect edges on the blended image using Laplacian
    gray = cv2.cvtColor(blended_np, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    edge_mask = np.clip(np.abs(laplacian) / 8.0, 0.0, 1.0)
    edge_mask = cv2.GaussianBlur(edge_mask, (3, 3), 1.0)
    
    # Define a feature-specific weight map
    # We give eyes, eyebrows, and lips extra boost, and beard a modest boost.
    feature_weight = (
        (masks["left_eye"] + masks["right_eye"]) * (1.0 + eye_boost) +
        (masks["left_eyebrow"] + masks["right_eyebrow"]) * 0.8 +
        masks["lips"] * 0.8 +
        masks["beard"] * 0.4
    )
    feature_weight = np.clip(feature_weight, 0.0, 2.5)
    
    # Construct sharpening map
    if np.sum(face_mask) > 0:
        # Strict edge-aware face sharpening
        sharpen_map = edge_mask * (face_mask + feature_weight)
    else:
        # Fallback to general edge-aware sharpening if no face mesh is found
        sharpen_map = edge_mask * (1.0 + texture_mask * 0.5)
        
    sharpen_map = np.clip(sharpen_map, 0.0, 2.5)
    
    # Apply sharpening strictly on edges
    blurred_blended = cv2.GaussianBlur(blended_np.astype(np.float32), (5, 5), 1.0)
    sharp_residual = blended_np.astype(np.float32) - blurred_blended
    
    final_float = blended_np.astype(np.float32) + sharpen_strength * sharp_residual * sharpen_map[:, :, np.newaxis]
    final_np = np.clip(final_float, 0.0, 255.0).astype(np.uint8)
    
    return Image.fromarray(final_np)
