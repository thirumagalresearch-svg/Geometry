from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Selected landmark indices for a stable facial frame.
LANDMARK_INDEX = {
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
    "nose_tip": 1,
}


def _to_pixel(lm, width: int, height: int) -> Tuple[int, int]:
    """Convert normalized landmark to pixel coordinates.
    
    Args:
        lm: MediaPipe landmark object with x, y attributes
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        Tuple of (x, y) pixel coordinates
    
    Raises:
        AttributeError: If landmark object missing x/y attributes
        ValueError: If width or height <= 0
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")
    if not hasattr(lm, 'x') or not hasattr(lm, 'y'):
        raise AttributeError("Landmark object missing x/y attributes")
    return int(lm.x * width), int(lm.y * height)


def extract_landmarks(image: Image.Image) -> Optional[Dict[str, Tuple[int, int]]]:
    """Extract a small set of face landmarks from a single image.
    
    Args:
        image: PIL Image in RGB format
    
    Returns:
        Dictionary mapping landmark names to (x, y) coordinates,
        or None if no face detected
    
    Raises:
        TypeError: If image is not PIL Image
        ValueError: If image is empty or invalid
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(image)}")
    
    try:
        mp_face_mesh = mp.solutions.face_mesh
        rgb = np.array(image, dtype=np.uint8)
        
        if rgb.size == 0:
            raise ValueError("Empty image array")
        
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            results = face_mesh.process(bgr)
        
        if not results.multi_face_landmarks:
            return None
        
        h, w = rgb.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        points = {}
        
        for name, idx in LANDMARK_INDEX.items():
            if idx >= len(landmarks):
                raise IndexError(f"Landmark index {idx} out of range (max: {len(landmarks)-1})")
            points[name] = _to_pixel(landmarks[idx], w, h)
        
        return points
    
    except Exception as e:
        # Re-raise with more context
        raise RuntimeError(f"MediaPipe landmark extraction failed: {str(e)}") from e


def extract_all_landmarks(image: Image.Image) -> Optional[np.ndarray]:
    """Extract all 468 mesh landmarks as a NumPy array of shape (468, 3) of coordinates (x, y, z).
    
    Args:
        image: PIL Image in RGB format
        
    Returns:
        NumPy array of (x, y, z) coordinates of shape (468, 3), or None if no face detected.
        Here x and y are in pixels, and z is the raw depth coordinate.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(image)}")
    
    try:
        mp_face_mesh = mp.solutions.face_mesh
        rgb = np.array(image, dtype=np.uint8)
        
        if rgb.size == 0:
            raise ValueError("Empty image array")
        
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            results = face_mesh.process(bgr)
        
        if not results.multi_face_landmarks:
            return None
        
        h, w = rgb.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Keep only the base 468 face mesh landmarks
        points = []
        for idx in range(min(468, len(landmarks))):
            lm = landmarks[idx]
            points.append((float(lm.x * w), float(lm.y * h), float(lm.z)))
            
        return np.array(points, dtype=np.float32)
        
    except Exception as e:
        raise RuntimeError(f"MediaPipe face mesh extraction failed: {str(e)}") from e
