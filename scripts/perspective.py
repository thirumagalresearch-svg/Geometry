import cv2
import numpy as np
from PIL import Image
from typing import Dict, Optional, Tuple, Any

from scripts.geometry import compute_face_mask, warp_face_mesh, rk4_warp_image


def rectify_perspective(
    image: Image.Image,
    depth_map: np.ndarray,
    landmarks: Optional[Dict[str, Tuple[int, int]]] = None,
    landmarks_all: Optional[np.ndarray] = None,
    shift: float = 1.2,
    warp_strength: float = 0.25,
    vertical_strength: float = 0.10,
    radial_k1: float = -0.05,
    smooth_percent: float = 0.30,
    return_intermediates: bool = False,
) -> Any:
    """Apply landmark-based facial mesh deformation and virtual camera reprojection.
    
    Args:
        image: PIL Image to rectify
        depth_map: Normalized depth map (0 to 1, where 1 is closest)
        landmarks: Optional named facial landmarks for centering
        landmarks_all: Optional NumPy array of shape (468, 2) of all face mesh landmarks
        shift: Virtual camera backward translation factor (meters)
        warp_strength: Warping intensity scale factor (0.0 to 1.0)
        vertical_strength: Relative Y-axis scaling intensity (0.0 to 1.0)
        radial_k1: Radial distortion coefficient (K1) for barrel distortion compensation
        smooth_percent: Unused (maintained for signature compatibility)
        return_intermediates: If True, returns (rectified, map_x, map_y, mask)
        
    Returns:
        Rectified PIL Image (or Tuple of PIL Image, map_x, map_y, mask if return_intermediates is True)
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(image)}")
        
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {w}x{h}")
        
    try:
        # Convert depth map to float32
        depth_map = depth_map.astype(np.float32)
        
        # Center of projection: use nose tip landmark if available, else image center
        if landmarks and "nose_tip" in landmarks:
            cx, cy = landmarks["nose_tip"]
        else:
            cx, cy = w / 2.0, h / 2.0
            
        # Grid coordinate maps
        uu, vv = np.meshgrid(np.arange(w), np.arange(h))
        
        # If no face mesh landmarks are available, fall back to global identity return
        if landmarks_all is None:
            print("[WARN] No face mesh landmarks available for deformable warp. Returning original image.")
            if return_intermediates:
                return image, uu.astype(np.float32), vv.astype(np.float32), np.ones((h, w), dtype=np.float32)
            return image

        # 1. Use anatomical z depth from landmarks to estimate physical distance
        Z_c = 0.45 # assumed camera-to-face center distance in meters
        
        # Normalize face width to compute depth scale dynamically
        x_coords = landmarks_all[:, 0]
        x_min_val, x_max_val = np.min(x_coords), np.max(x_coords)
        norm_width = (x_max_val - x_min_val) / w
        if norm_width <= 0:
            norm_width = 0.35 # fallback default
            
        S = 0.15 / norm_width # scale factor (assumes average human face width is 15cm)
        
        # Physical depth of each landmark
        # MediaPipe raw z coordinates represent depth relative to face center
        z_old_pts = Z_c + landmarks_all[:, 2] * S
        z_old_pts = np.clip(z_old_pts, 0.1, 10.0).astype(np.float32)
        z_avg = Z_c
        
        # 2. Virtual Camera setup (assumed 60 degree FoV camera)
        fov_deg = 60.0
        f_selfie = w / (2 * np.tan(np.radians(fov_deg) / 2))
        
        # 3. Back-project 2D landmarks to 3D space
        X = (landmarks_all[:, 0] - cx) / f_selfie * z_old_pts
        Y = (landmarks_all[:, 1] - cy) / f_selfie * z_old_pts
        
        # 4. Apply Radial Distortion Correction to landmarks (Barrel compensation)
        xx = (landmarks_all[:, 0] - cx) / f_selfie
        yy = (landmarks_all[:, 1] - cy) / f_selfie
        r2 = xx**2 + yy**2
        
        xx_radial = xx * (1.0 + radial_k1 * r2)
        yy_radial = yy * (1.0 + radial_k1 * r2)
        X_radial = xx_radial * z_old_pts
        Y_radial = yy_radial * z_old_pts
        
        # 5. Simulate Portrait Lens Projection (Shift camera back by dz)
        dz = shift
        f_portrait = f_selfie * (z_avg + dz) / z_avg
        
        # Reproject 3D points
        u_proj = cx + f_portrait * X_radial / (z_old_pts + dz)
        v_proj = cy + f_portrait * Y_radial / (z_old_pts + dz)
        
        # 6. Compute boundary-constrained displacement weights (decay weight)
        landmarks_all_2d = landmarks_all[:, :2].astype(np.float32)
        
        # Center of face is the nose tip (landmark index 1)
        nose_tip = landmarks_all_2d[1]
        dists = np.linalg.norm(landmarks_all_2d - nose_tip, axis=1)
        
        # Find characteristic face radius using bounding box of landmarks
        x_min, y_min = np.min(landmarks_all_2d, axis=0)
        x_max, y_max = np.max(landmarks_all_2d, axis=0)
        face_width = x_max - x_min
        face_height = y_max - y_min
        R_face = 0.5 * (face_width + face_height)
        R_face = max(1.0, R_face)
        
        r_dist = dists / R_face
        
        # Gaussian falloff centered at the nose tip (sigma=0.35 concentrates on center-face)
        sigma = 0.35
        weights = np.exp(-0.5 * (r_dist / sigma)**2)
        
        # Linearly decay to exactly 0 between r_dist=0.4 and r_dist=0.6
        # This guarantees landmarks at r_dist >= 0.6 have exactly 0 displacement.
        cutoff_start = 0.4
        cutoff_end = 0.6
        ramp = np.clip((cutoff_end - r_dist) / (cutoff_end - cutoff_start), 0.0, 1.0)
        weights = (weights * ramp).astype(np.float32)
        
        # 7. Apply Warp Strength, Y-axis scale constraint, and Boundary weights
        du = u_proj - landmarks_all[:, 0]
        dv = v_proj - landmarks_all[:, 1]
        
        # Force vertical scale constraint to exactly 0 to prevent all vertical stretching/forehead elongation
        target_pts_x = landmarks_all[:, 0] + du * warp_strength * weights
        target_pts_y = landmarks_all[:, 1]  # Zero vertical scaling (strictly horizontal adjustment)
        target_pts = np.stack([target_pts_x, target_pts_y], axis=1).astype(np.float32)
        
        # 8. Compute piecewise affine warped coordinates mapping
        map_x_pa, face_mask = warp_face_mesh(uu.astype(np.float32), landmarks_all_2d, target_pts)
        map_y_pa, _ = warp_face_mesh(vv.astype(np.float32), landmarks_all_2d, target_pts)
        
        # 9. Warp the image and mask using Runge-Kutta 4th order path integration of the velocity field
        rgb = np.array(image, dtype=np.uint8)
        warped_face = rk4_warp_image(rgb, map_x_pa, map_y_pa, num_steps=4)
        
        # Integrate face mask using RK4
        mask = rk4_warp_image(face_mask, map_x_pa, map_y_pa, num_steps=4)
        # Normalize mask
        mask = mask.astype(np.float32) / 255.0 if mask.dtype == np.uint8 else mask
        
        if return_intermediates:
            map_x = map_x_pa
            map_y = map_y_pa
        else:
            map_x = uu.astype(np.float32)
            map_y = vv.astype(np.float32)

        # 10. Blend using cv2.seamlessClone inside the face ROI
        import cv2
        active_indices = np.where(weights > 0.0)[0]
        if len(active_indices) >= 3:
            active_pts = landmarks_all_2d[active_indices].astype(np.int32)
            hull = cv2.convexHull(active_pts)
            # Create clone mask
            clone_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(clone_mask, hull, 255)
            
            # Erode slightly to stay within boundary
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            clone_mask = cv2.erode(clone_mask, kernel, iterations=1)
            
            # Bounding box center for cloning
            x_b, y_b, w_b, h_b = cv2.boundingRect(hull)
            center_clone = (x_b + w_b // 2, y_b + h_b // 2)
            
            try:
                # Blend the warped face into the original image
                cloned_rgb = cv2.seamlessClone(warped_face, rgb, clone_mask, center_clone, cv2.NORMAL_CLONE)
                # Keep original pixels strictly outside the clone mask
                final_rgb = rgb.copy()
                final_rgb[clone_mask == 255] = cloned_rgb[clone_mask == 255]
            except Exception as e_clone:
                print(f"[WARN] cv2.seamlessClone failed: {str(e_clone)}. Falling back to smooth linear blend.")
                mask_3d = mask[:, :, np.newaxis]
                final_rgb = (warped_face.astype(np.float32) * mask_3d + rgb.astype(np.float32) * (1.0 - mask_3d)).astype(np.uint8)
        else:
            mask_3d = mask[:, :, np.newaxis]
            final_rgb = (warped_face.astype(np.float32) * mask_3d + rgb.astype(np.float32) * (1.0 - mask_3d)).astype(np.uint8)
            
        rectified_image = Image.fromarray(final_rgb)
        
        if return_intermediates:
            return rectified_image, map_x, map_y, mask
            
        return rectified_image
        
    except Exception as e:
        print(f"[WARN] Landmark-based perspective rectification failed: {str(e)}")
        import traceback
        traceback.print_exc()
        if return_intermediates:
            return image, uu.astype(np.float32), vv.astype(np.float32), np.ones((h, w), dtype=np.float32)
        return image
