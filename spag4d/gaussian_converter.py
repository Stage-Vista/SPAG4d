# spag4d/gaussian_converter.py
"""
Convert equirectangular panorama + depth to 3D Gaussians.

This implements the SPAG (Spherical Panorama to Gaussians) algorithm
with latitude-aware anisotropic scaling.
"""

import torch
import math
from typing import Optional

from .spherical_grid import SphericalGrid, rotation_matrix_to_quaternion


def equirect_to_gaussians(
    image: torch.Tensor,
    depth: torch.Tensor,
    grid: SphericalGrid,
    scale_factor: float = 1.5,
    thickness_ratio: float = 0.1,
    depth_min: float = 0.1,
    depth_max: float = 100.0,
    pole_rows: int = 3,
    default_opacity: float = 0.95,
    validity_mask: Optional[torch.Tensor] = None
) -> dict:
    """
    Convert equirectangular panorama with depth to 3D Gaussians.
    
    Args:
        image: RGB image [H, W, 3] uint8 or float [0,1]
        depth: Depth map [H, W] in meters
        grid: Precomputed SphericalGrid
        scale_factor: Gaussian scale multiplier (larger = more overlap)
        thickness_ratio: Radial thickness as fraction of tangent scales
        depth_min: Minimum valid depth
        depth_max: Maximum valid depth
        pole_rows: Rows to exclude at top/bottom poles
        default_opacity: Opacity for all Gaussians
        validity_mask: Optional [H, W] mask from depth model (0-1 or bool)
    
    Returns:
        Dict with:
            means: [N, 3] 3D positions (Y-up frame)
            scales: [N, 3] Gaussian scales (azimuth, elevation, normal)
            quats: [N, 4] Quaternions in XYZW order
            colors: [N, 3] RGB colors [0, 1]
            opacities: [N, 1] Opacity values
    """
    device = grid.device
    H, W = grid.original_H, grid.original_W
    stride = grid.stride
    
    # Convert image to float if needed
    if image.dtype == torch.uint8:
        colors = image.float() / 255.0
    else:
        colors = image.clone()
    
    # Downsample image to match grid
    H_grid, W_grid = grid.theta.shape
    if colors.shape[0] != H_grid or colors.shape[1] != W_grid:
        # Use strided sampling (not interpolation for speed)
        colors = colors[stride//2::stride, stride//2::stride]
    
    # Ensure depth matches grid dimensions
    if depth.shape[0] != H_grid or depth.shape[1] != W_grid:
        depth = depth[stride//2::stride, stride//2::stride]
    
    # Downsample validity_mask to match grid if provided
    if validity_mask is not None:
        if validity_mask.shape[0] != H_grid or validity_mask.shape[1] != W_grid:
            validity_mask = validity_mask[stride//2::stride, stride//2::stride]
    
    # ─────────────────────────────────────────────────────────────────
    # Validity Mask
    # ─────────────────────────────────────────────────────────────────
    valid_mask = (depth > depth_min) & (depth < depth_max)
    
    # Apply learned validity mask if provided
    if validity_mask is not None:
        # Convert to bool if float (threshold at 0.5)
        if validity_mask.dtype == torch.float32 or validity_mask.dtype == torch.float16:
            valid_mask = valid_mask & (validity_mask > 0.5)
        else:
            valid_mask = valid_mask & validity_mask.bool()
    
    # Exclude poles
    if pole_rows > 0:
        pole_mask = torch.ones_like(valid_mask, dtype=torch.bool)
        pole_mask[:pole_rows, :] = False
        pole_mask[-pole_rows:, :] = False
        valid_mask = valid_mask & pole_mask
    
    # ─────────────────────────────────────────────────────────────────
    # 3D Positions: P = depth * r̂
    # ─────────────────────────────────────────────────────────────────
    means = depth.unsqueeze(-1) * grid.rhat
    
    # ─────────────────────────────────────────────────────────────────
    # Anisotropic Scale (latitude-aware)
    # ─────────────────────────────────────────────────────────────────
    # Angular extent per pixel
    delta_theta = 2 * math.pi / W  # Horizontal angular width
    delta_phi = math.pi / H        # Vertical angular height
    
    # sin(φ) compensates for ERP distortion at poles
    sin_phi = torch.sin(grid.phi).clamp(min=0.01)  # Avoid zero at poles
    
    # Tangent plane footprint at distance d
    # s_azimuth: horizontal extent (varies with latitude)
    s_azimuth = scale_factor * depth * sin_phi * delta_theta * stride
    
    # s_elevation: vertical extent (constant with latitude in ERP)
    s_elevation = scale_factor * depth * delta_phi * stride
    
    # s_normal: thin in radial direction
    s_normal = torch.minimum(s_azimuth, s_elevation) * thickness_ratio
    
    # Stack: [azimuth, elevation, normal]
    scales = torch.stack([s_azimuth, s_elevation, s_normal], dim=-1)
    
    # ─────────────────────────────────────────────────────────────────
    # Rotation Matrix → Quaternion
    # ─────────────────────────────────────────────────────────────────
    # Build rotation matrix R where columns are [right, up, normal]
    # Normal points inward (toward camera), same as -rhat
    normal = -grid.rhat
    right = grid.tangent_right
    up = grid.tangent_up
    
    # R = [right | up | normal] as column vectors
    R = torch.stack([right, up, normal], dim=-1)  # [H, W, 3, 3]
    
    # Convert to quaternion
    quats = rotation_matrix_to_quaternion(R)  # [H, W, 4] in XYZW
    
    # ─────────────────────────────────────────────────────────────────
    # Flatten and filter by validity mask
    # ─────────────────────────────────────────────────────────────────
    valid_flat = valid_mask.flatten()
    
    means_flat = means.reshape(-1, 3)[valid_flat]
    scales_flat = scales.reshape(-1, 3)[valid_flat]
    quats_flat = quats.reshape(-1, 4)[valid_flat]
    colors_flat = colors.reshape(-1, 3)[valid_flat]
    
    N = means_flat.shape[0]
    opacities_flat = torch.full((N, 1), default_opacity, device=device)
    
    return {
        'means': means_flat,
        'scales': scales_flat,
        'quats': quats_flat,
        'colors': colors_flat,
        'opacities': opacities_flat
    }


def filter_sky(
    gaussians: dict,
    sky_threshold: float = 80.0
) -> dict:
    """
    Remove Gaussians beyond sky threshold distance.
    
    Args:
        gaussians: Gaussian dict from equirect_to_gaussians
        sky_threshold: Max distance in meters
    
    Returns:
        Filtered Gaussian dict
    """
    distances = gaussians['means'].norm(dim=-1)
    valid = distances < sky_threshold
    
    return {
        'means': gaussians['means'][valid],
        'scales': gaussians['scales'][valid],
        'quats': gaussians['quats'][valid],
        'colors': gaussians['colors'][valid],
        'opacities': gaussians['opacities'][valid]
    }
