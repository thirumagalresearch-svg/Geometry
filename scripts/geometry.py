"""3D geometry utilities for camera models and projection."""
from typing import Tuple

import numpy as np
import torch


def intrinsics_from_fov(
    height: int, width: int, fov_deg: float = 50.0
) -> np.ndarray:
    """Create camera intrinsic matrix from field-of-view."""
    focal = width / (2 * np.tan(np.radians(fov_deg) / 2))
    cx, cy = width / 2.0, height / 2.0
    K = np.array(
        [[focal, 0, cx], [0, focal, cy], [0, 0, 1]],
        dtype=np.float32,
    )
    return K


def depth_to_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Convert depth map to 3D point cloud."""
    h, w = depth.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    
    K_inv = np.linalg.inv(intrinsics)
    xy1 = np.stack([xx, yy, np.ones_like(xx)], axis=-1).reshape(-1, 3).T
    
    points_3d = K_inv @ xy1 * depth.reshape(-1)
    return points_3d.T


def estimate_camera_extrinsics(
    landmarks_2d: dict,
    intrinsics: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate camera rotation and translation from 2D landmarks.
    
    Placeholder for future integration with cv2.solvePnP or ML-based methods.
    """
    # Placeholder: Return identity transform
    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    return R, t


def project_points(
    points_3d: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Project 3D points to 2D image coordinates."""
    points_cam = (R @ points_3d.T + t.reshape(3, 1)).T
    points_2d = (K @ points_cam.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]
    return points_2d


def compute_face_mask(image_shape: Tuple[int, int], landmarks_all: np.ndarray) -> np.ndarray:
    """Create a smoothed face mask from the convex hull of 468 landmarks.
    
    Args:
        image_shape: Tuple of (height, width) of the image
        landmarks_all: NumPy array of shape (468, 2) or (468, 3) of pixel coordinates
        
    Returns:
        NumPy float32 array of shape (height, width) between 0.0 and 1.0
    """
    import cv2
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Extract only 2D coordinates (x, y)
    pts_2d = landmarks_all[:, :2].astype(np.int32)
    
    # Compute convex hull of all landmarks
    hull = cv2.convexHull(pts_2d)
    
    # Fill the polygon on the mask
    cv2.fillConvexPoly(mask, hull, 255)
    
    # Smooth the mask edges tightly (3% of bounding box) to prevent hair/background stretching
    x, y, w_box, h_box = cv2.boundingRect(hull)
    ksize = int(max(w_box, h_box) * 0.03) | 1
    if ksize >= 3:
        mask_blurred = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (ksize, ksize), 0)
    else:
        mask_blurred = mask.astype(np.float32) / 255.0
        
    return mask_blurred


def rk4_warp_image(image_np: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, num_steps: int = 4) -> np.ndarray:
    """Remap image using Runge-Kutta 4th order path integration of the displacement flow field.
    
    This ensures smooth diffeomorphic warping, preventing grid overlaps or folding
    even under intense displacement fields.
    """
    import cv2
    h, w = image_np.shape[:2]
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    
    # Compute constant-in-time velocity field (displacement map)
    vx = (map_x - uu).astype(np.float32)
    vy = (map_y - vv).astype(np.float32)
    
    # State coordinates for integration, initialized to identity grid
    curr_x = uu.astype(np.float32)
    curr_y = vv.astype(np.float32)
    
    dt = 1.0 / num_steps
    for _ in range(num_steps):
        # Sample velocity field at fractional coordinates using bilinear interpolation
        # cv2.remap is highly optimized for this operation
        k1_x = cv2.remap(vx, curr_x, curr_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        k1_y = cv2.remap(vy, curr_x, curr_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        ax = curr_x + 0.5 * dt * k1_x
        ay = curr_y + 0.5 * dt * k1_y
        k2_x = cv2.remap(vx, ax, ay, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        k2_y = cv2.remap(vy, ax, ay, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        bx = curr_x + 0.5 * dt * k2_x
        by = curr_y + 0.5 * dt * k2_y
        k3_x = cv2.remap(vx, bx, by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        k3_y = cv2.remap(vy, bx, by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        cx = curr_x + dt * k3_x
        cy = curr_y + dt * k3_y
        k4_x = cv2.remap(vx, cx, cy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        k4_y = cv2.remap(vy, cx, cy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        curr_x = curr_x + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        curr_y = curr_y + (dt / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
        
    # Remap final image using integrated coordinate paths
    warped = cv2.remap(image_np, curr_x, curr_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return warped


def generate_debug_vis(
    image: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    mask: np.ndarray,
    landmarks_before: dict,
    landmarks_after: dict,
) -> np.ndarray:
    """Generate a 4-panel debug visualization image.
    
    Args:
        image: Original RGB NumPy image of shape (h, w, 3)
        map_x: Warping map X coordinate array
        map_y: Warping map Y coordinate array
        mask: Smooth face mask of shape (h, w) between 0 and 1
        landmarks_before: Original key points dict
        landmarks_after: Rectified key points dict
        
    Returns:
        RGB NumPy image of shape (h * 2, w * 2, 3) containing 4 subplots
    """
    import cv2
    h, w = image.shape[:2]
    
    # 1. Panel 1: Face ROI Mask Highlighted
    overlay = image.copy()
    overlay[mask > 0.05] = (overlay[mask > 0.05] * 0.5 + np.array([30, 144, 255]) * 0.5).astype(np.uint8)
    p1 = overlay
    cv2.putText(p1, "1. Facial ROI Mask", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # 2. Panel 2: Warp Mesh Grid with Delaunay Triangulation overlay
    grid_img = np.zeros((h, w, 3), dtype=np.uint8)
    grid_step = max(h, w) // 25
    for x in range(0, w, grid_step):
        cv2.line(grid_img, (x, 0), (x, h), (0, 255, 0), 1)
    for y in range(0, h, grid_step):
        cv2.line(grid_img, (0, y), (w, y), (0, 255, 0), 1)
        
    # Warp grid image
    warped_grid = cv2.remap(
        grid_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    # Warp original image for overlay base
    warped_img = rk4_warp_image(image, map_x, map_y, num_steps=4)
    
    p2 = warped_img.copy()
    grid_mask = (warped_grid > 0).any(axis=-1)
    p2[grid_mask] = (p2[grid_mask] * 0.5 + warped_grid[grid_mask] * 0.5).astype(np.uint8)
    
    # Render Delaunay triangles overlay on Panel 2
    try:
        subdiv = cv2.Subdiv2D((0, 0, w, h))
        pts = list(landmarks_after.values())
        for pt in pts:
            subdiv.insert((float(pt[0]), float(pt[1])))
        triangles = subdiv.getTriangleList()
        for t in triangles:
            pt1 = (int(t[0]), int(t[1]))
            pt2 = (int(t[2]), int(t[3]))
            pt3 = (int(t[4]), int(t[5]))
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h and
                0 <= pt3[0] < w and 0 <= pt3[1] < h):
                cv2.line(p2, pt1, pt2, (100, 255, 100), 1, cv2.LINE_AA)
                cv2.line(p2, pt2, pt3, (100, 255, 100), 1, cv2.LINE_AA)
                cv2.line(p2, pt3, pt1, (100, 255, 100), 1, cv2.LINE_AA)
    except Exception as e_tri:
        print(f"[WARN] Failed to render triangles on debug Panel 2: {e_tri}")
        
    cv2.putText(p2, "2. Warped Mesh Grid & Triangles", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # 3. Panel 3: Deformation Vectors
    p3 = image.copy()
    step = max(h, w) // 20
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    dx = uu - map_x
    dy = vv - map_y
    
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if mask[y, x] > 0.05:
                vx = int(dx[y, x] * 3.0)  # Magnify vector for visibility
                vy = int(dy[y, x] * 3.0)
                if abs(vx) > 1 or abs(vy) > 1:
                    cv2.arrowedLine(p3, (x, y), (x + vx, y + vy), (0, 0, 255), 2, tipLength=0.3)
                    
    cv2.putText(p3, "3. Deformation Vectors (3x)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # 4. Panel 4: Before/After Landmarks
    p4 = warped_img.copy()
    for name, pt in landmarks_before.items():
        cv2.circle(p4, pt, radius=8, color=(0, 0, 255), thickness=-1)
        cv2.circle(p4, pt, radius=9, color=(255, 255, 255), thickness=1)
        
    for name, pt in landmarks_after.items():
        cv2.circle(p4, pt, radius=8, color=(0, 255, 0), thickness=-1)
        cv2.circle(p4, pt, radius=9, color=(0, 0, 0), thickness=1)
        
    cv2.putText(p4, "4. Face Alignment Comparison", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # Combine into 2x2 grid
    top_row = np.hstack([p1, p2])
    bottom_row = np.hstack([p3, p4])
    debug_vis = np.vstack([top_row, bottom_row])
    
    return debug_vis


def warp_face_mesh(
    image_np: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp the face region of the image using Delaunay Triangulation piecewise affine warping.
    
    Args:
        image_np: Original image of shape (h, w, C) or (h, w)
        src_pts: Source landmark points of shape (N, 2) or (N, 3)
        dst_pts: Target landmark points of shape (N, 2) or (N, 3)
        
    Returns:
        Tuple of (warped_image, face_mask_blurred)
    """
    import cv2
    h, w = image_np.shape[:2]
    
    # Ensure points are 2D coordinates (x, y)
    src_pts_2d = src_pts[:, :2]
    dst_pts_2d = dst_pts[:, :2]
    
    # Bounding box coordinates for subdivision insertion
    src_pts_clipped = np.clip(src_pts_2d, [0, 0], [w - 1, h - 1]).astype(np.float32)
    dst_pts_clipped = np.clip(dst_pts_2d, [0, 0], [w - 1, h - 1]).astype(np.float32)
    
    # Avoid exact duplicate points for Subdiv2D
    cleaned_src = []
    seen = set()
    for pt in src_pts_clipped:
        pt_tuple = (float(pt[0]), float(pt[1]))
        if pt_tuple in seen:
            pt_tuple = (pt_tuple[0] + np.random.uniform(-0.01, 0.01), pt_tuple[1] + np.random.uniform(-0.01, 0.01))
        seen.add(pt_tuple)
        cleaned_src.append(pt_tuple)
    cleaned_src = np.array(cleaned_src, dtype=np.float32)

    # 1. Compute Delaunay Triangulation on source points
    subdiv = cv2.Subdiv2D((0, 0, w, h))
    for pt in cleaned_src:
        subdiv.insert((float(pt[0]), float(pt[1])))
        
    triangle_list = subdiv.getTriangleList()
    
    # Create output canvas
    warped = image_np.copy()
    face_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Helper to find indices of vertices
    def get_landmark_idx(pt):
        dists = np.sum((cleaned_src - pt)**2, axis=1)
        return np.argmin(dists)

    for t in triangle_list:
        p1 = (t[0], t[1])
        p2 = (t[2], t[3])
        p3 = (t[4], t[5])
        
        # Check if triangle is inside image boundary
        if not (0 <= p1[0] < w and 0 <= p1[1] < h and
                0 <= p2[0] < w and 0 <= p2[1] < h and
                0 <= p3[0] < w and 0 <= p3[1] < h):
            continue
            
        # Get landmark indices
        idx1 = get_landmark_idx(p1)
        idx2 = get_landmark_idx(p2)
        idx3 = get_landmark_idx(p3)
        
        # Source triangle vertices
        tri_src = np.array([cleaned_src[idx1], cleaned_src[idx2], cleaned_src[idx3]], dtype=np.float32)
        
        # Target triangle vertices
        tri_dst = np.array([dst_pts_clipped[idx1], dst_pts_clipped[idx2], dst_pts_clipped[idx3]], dtype=np.float32)
        
        # Bounding box of target triangle
        x, y, w_box, h_box = cv2.boundingRect(tri_dst)
        if w_box <= 0 or h_box <= 0:
            continue
            
        # Create mask for target triangle
        mask = np.zeros((h_box, w_box), dtype=np.uint8)
        tri_dst_local = tri_dst - np.array([x, y])
        cv2.fillConvexPoly(mask, tri_dst_local.astype(np.int32), 255)
        
        # Compute affine transform from target (local) to source
        # Note: mapping is from output (target) to input (source) to warp pixel coordinate grid
        M = cv2.getAffineTransform(np.float32(tri_dst_local), np.float32(tri_src))
        
        # Warp the bounding box region
        # Use bilinear/cubic interpolation depending on image channel structure
        interp = cv2.INTER_CUBIC if len(image_np.shape) == 3 else cv2.INTER_LINEAR
        warped_box = cv2.warpAffine(
            image_np,
            M,
            (w_box, h_box),
            flags=interp,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        # Copy warped pixels using the mask
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + w_box), min(h, y + h_box)
        
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 0 or box_h <= 0:
            continue
            
        local_mask_slice = mask[0:box_h, 0:box_w]
        warped_box_slice = warped_box[0:box_h, 0:box_w]
        
        # Write to warped output and update mask
        if len(image_np.shape) == 3:
            warped[y1:y2, x1:x2][local_mask_slice == 255] = warped_box_slice[local_mask_slice == 255]
        else:
            warped[y1:y2, x1:x2][local_mask_slice == 255] = warped_box_slice[local_mask_slice == 255]
            
        face_mask[y1:y2, x1:x2][local_mask_slice == 255] = 255

    # Smooth the face mask to ensure a seamless boundary blend
    x, y, w_box, h_box = cv2.boundingRect(src_pts_clipped.astype(np.int32))
    ksize = int(max(w_box, h_box) * 0.03) | 1
    if ksize >= 3:
        face_mask_blurred = cv2.GaussianBlur(face_mask.astype(np.float32) / 255.0, (ksize, ksize), 0)
    else:
        face_mask_blurred = face_mask.astype(np.float32) / 255.0
        
    return warped, face_mask_blurred
