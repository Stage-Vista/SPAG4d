# spag4d/core.py
"""
Main SPAG4D class that orchestrates the conversion pipeline.
"""

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Literal, Union, List
import time

from PIL import Image, ImageOps


def _rotation_matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion (XYZW order).

    Uses Shepperd's method for numerical stability.

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion [x, y, z, w]
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)  # Normalize


@dataclass
class ConversionResult:
    """Result of SPAG-4D conversion."""
    output_path: str
    splat_count: int
    file_size: int
    processing_time: float
    depth_range: tuple


class SPAG4D:
    """
    SPAG-4D: 360° Panorama to Gaussian Splat converter.
    
    Uses PanDA or DAP for native equirectangular depth estimation
    and SPAG algorithm for Gaussian splat generation.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        use_mock_dap: bool = False,
        depth_model: str = "panda",  # "panda", "da3", "dap", or "mock"
        use_guided_filter: bool = True,
        guided_filter_radius: int = 8,
        guided_filter_eps: float = 1e-4,
        use_sharp_refinement: bool = True,
        sharp_model_path: Optional[str] = None,
        sharp_cubemap_size: int = 1536,
        sharp_projection_mode: str = "cubemap",  # "cubemap" or "icosahedral"
    ):
        """
        Initialize SPAG4D converter.

        Args:
            device: Device for computation ("cuda", "cpu", "mps")
            model_path: Optional explicit path to depth model weights
            use_mock_dap: Use mock DAP model (for testing without weights)
            depth_model: Depth model to use: "panda" (default), "da3", "dap", or "mock"
            use_guided_filter: Enable RGB-guided depth edge refinement
            guided_filter_radius: Guided filter radius (larger = smoother)
            guided_filter_eps: Guided filter regularization (smaller = sharper edges)
            use_sharp_refinement: Enable SHARP attribute refinement (default: True)
            sharp_model_path: Optional path to SHARP weights
            sharp_cubemap_size: Cubemap face size for SHARP (must be multiple of 384)
            sharp_projection_mode: "cubemap" (6 faces) or "icosahedral" (20 faces)
        """
        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )

        # Handle legacy use_mock_dap parameter
        if use_mock_dap:
            depth_model = "mock"

        # Load depth model
        self.depth_model_name = depth_model
        if depth_model == "mock":
            from .dap_model import MockDAPModel
            self.dap = MockDAPModel.load(device=self.device)
        elif depth_model == "panda":
            from .panda_model import PanDAModel
            self.dap = PanDAModel.load(model_path, device=self.device)
        elif depth_model == "da3":
            from .da3_model import DA3Model
            self.dap = DA3Model.load(device=self.device)
        else:  # "dap" or fallback
            from .dap_model import DAPModel
            self.dap = DAPModel.load(model_path, device=self.device)

        # Guided depth filter (sharpens depth edges using RGB guide)
        self.guided_refiner = None
        if use_guided_filter:
            from .depth_refiner import GuidedDepthRefiner
            self.guided_refiner = GuidedDepthRefiner(
                radius=guided_filter_radius,
                eps=guided_filter_eps,
            )

        # SHARP refinement (enabled by default for maximum quality)
        self.sharp_refiner = None
        if use_sharp_refinement:
            try:
                from .sharp_refiner import SHARPRefiner
                self.sharp_refiner = SHARPRefiner(
                    device=self.device,
                    cubemap_size=sharp_cubemap_size,
                    refine_colors=True,
                    projection_mode=sharp_projection_mode,
                )
                # Preload if path provided, otherwise lazy load
                if sharp_model_path:
                    self.sharp_refiner.load_model(sharp_model_path)
            except ImportError:
                import warnings
                warnings.warn(
                    "SHARP not available. Install with: "
                    "pip install git+https://github.com/apple/ml-sharp.git "
                    "Falling back to geometric-only Gaussians."
                )
        self.sharp_cubemap_size = sharp_cubemap_size
        self.sharp_projection_mode = sharp_projection_mode
        
        # Cache for spherical grids (by resolution + stride)
        self._grid_cache = {}
    
    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        stride: int = 2,
        scale_factor: float = 1.5,
        thickness_ratio: float = 0.1,
        global_scale: float = 1.0,
        depth_min: float = 0.1,
        depth_max: float = 100.0,
        sky_threshold: float = 80.0,
        sh_degree: int = 0,
        output_format: Literal["ply", "splat"] = "ply",

        depth_gradient_threshold: float = 0.5,
        force_erp: bool = False,
        depth_preview_path: Optional[Union[str, Path]] = None,
        precomputed_depth: Optional[Union[str, Path, torch.Tensor]] = None,
        **kwargs
    ) -> ConversionResult:
        """
        Convert equirectangular panorama to Gaussian splat.

        Args:
            input_path: Path to input ERP image
            output_path: Path for output file
            stride: Spatial downsampling factor (1, 2, 4, 8)
            scale_factor: Gaussian scale multiplier
            thickness_ratio: Radial thickness as fraction of tangent scales
            global_scale: Manual depth scale multiplier (for scale correction)
            depth_min: Minimum valid depth in meters
            depth_max: Maximum valid depth in meters
            sky_threshold: Depth above this is treated as sky (0 to disable)
            sh_degree: Spherical harmonics degree (0 or 3)
            output_format: Output format ("ply" or "splat")
            force_erp: Process even if aspect ratio isn't 2:1
            precomputed_depth: Optional pre-computed metric depth map (EXR path,
                numpy array, or torch tensor [H, W] in metres). When provided,
                skips the internal depth model entirely, which is essential when
                the camera poses (cameras.npz) were generated with a metric depth
                model (e.g. DAP) so the splat scale matches the pose scale.

        Returns:
            ConversionResult with output details
        """
        from .spherical_grid import create_spherical_grid
        from .gaussian_converter import equirect_to_gaussians, filter_sky
        from .ply_writer import save_ply_gsplat
        from .splat_writer import save_splat
        
        start_time = time.time()
        
        # Load and validate image
        img = Image.open(input_path).convert('RGB')
        img = ImageOps.exif_transpose(img)  # Handle EXIF orientation
        
        W, H = img.size
        aspect = W / H
        
        if not (1.9 < aspect < 2.1) and not force_erp:
            raise ValueError(
                f"Image aspect ratio {aspect:.2f} is not 2:1. "
                "Use --force-erp to process anyway."
            )
            
        image_tensor = torch.from_numpy(np.array(img)).to(self.device)
        
        # Get or create spherical grid
        grid_key = (H, W, stride)
        if grid_key not in self._grid_cache:
            self._grid_cache[grid_key] = create_spherical_grid(
                H, W, self.device, stride=stride
            )
        grid = self._grid_cache[grid_key]
        
        # Estimate depth — use precomputed metric depth if provided, otherwise run model
        if precomputed_depth is not None:
            if isinstance(precomputed_depth, (str, Path)):
                import cv2 as _cv2
                _os = __import__("os")
                _os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
                depth_np = _cv2.imread(str(precomputed_depth), _cv2.IMREAD_ANYDEPTH | _cv2.IMREAD_ANYCOLOR)
                if depth_np is None:
                    raise ValueError(f"Failed to load depth map: {precomputed_depth}")
                if depth_np.ndim == 3:
                    depth_np = depth_np[:, :, 0]
                depth = torch.from_numpy(depth_np.astype(np.float32)).to(self.device)
            elif isinstance(precomputed_depth, np.ndarray):
                depth = torch.from_numpy(precomputed_depth.astype(np.float32)).to(self.device)
            else:
                depth = precomputed_depth.to(self.device)
            # Resize depth to match image if resolutions differ (e.g. depth saved
            # at 1440×720 but image is 5760×2880 from S3PO super-resolution).
            if depth.shape != (H, W):
                import torch.nn.functional as _F
                depth = _F.interpolate(
                    depth.unsqueeze(0).unsqueeze(0),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                print(f"[SPAG4D] Resized precomputed depth to {W}x{H}")
            validity_mask = None
            print(f"[SPAG4D] Using precomputed depth: range [{depth.min():.2f}, {depth.max():.2f}] m")
        else:
            # Estimate depth with depth model (PanDA, DAP, or mock)
            with torch.inference_mode():
                depth, validity_mask = self.dap.predict(image_tensor)
        
        # Apply RGB-guided depth edge refinement
        if self.guided_refiner is not None:
            guided_strength = kwargs.get('guided_strength', 1.0)
            if guided_strength > 0:
                img_float = image_tensor.float() / 255.0 if image_tensor.dtype == torch.uint8 else image_tensor.float()
                depth = self.guided_refiner.refine(depth, img_float, strength=guided_strength)
        
        # Save depth preview if requested
        if depth_preview_path:
            try:
                import cv2
                
                # Normalize depth for visualization (log scale for better dynamic range)
                depth_np = depth.cpu().numpy()
                depth_log = np.log1p(depth_np)
                
                # Robust min/max to avoid outliers
                d_min = np.percentile(depth_log, 1)
                d_max = np.percentile(depth_log, 99)
                
                if d_max > d_min:
                    depth_norm = np.clip((depth_log - d_min) / (d_max - d_min), 0, 1)
                    depth_norm = (depth_norm * 255).astype(np.uint8)
                else:
                    depth_norm = np.zeros_like(depth_log, dtype=np.uint8)
                
                # Apply colormap (INFERNO is excellent for depth)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
                cv2.imwrite(str(depth_preview_path), depth_color)
            except Exception as e:
                print(f"Failed to save depth preview: {e}")
        
        # Apply global scale correction
        depth = depth * global_scale
        
        # Apply sky threshold as fallback (if model doesn't provide mask)
        if validity_mask is None and sky_threshold > 0:
            validity_mask = (depth <= sky_threshold).float()
        elif sky_threshold > 0:
            # Combine learned mask with sky threshold
            validity_mask = validity_mask * (depth <= sky_threshold).float()
        
        # SHARP Refinement (enabled by default when refiner is available)
        refined_attrs = None
        use_sharp = kwargs.get('use_sharp_refinement', self.sharp_refiner is not None)

        if use_sharp:
            # Check if resolution changed
            current_size = self.sharp_refiner.cubemap_size if self.sharp_refiner else 0
            target_size = kwargs.get('sharp_cubemap_size', self.sharp_cubemap_size)
            
            # Re-instantiate if needed (or if not yet created)
            if self.sharp_refiner is None or current_size != target_size:
                try:
                    from .sharp_refiner import SHARPRefiner
                    if self.sharp_refiner:
                        print(f"Switching SHARP resolution: {current_size} -> {target_size}")
                    
                    self.sharp_refiner = SHARPRefiner(
                        device=self.device,
                        cubemap_size=target_size,
                        refine_colors=True,
                        projection_mode=self.sharp_projection_mode,
                    )
                    # Update default size
                    self.sharp_cubemap_size = target_size
                except ImportError:
                    use_sharp = False
            
            # Ensure model is loaded (no-op if already loaded)
            if self.sharp_refiner is not None:
                self.sharp_refiner.load_model()
            else:
                use_sharp = False
        
        # Run SHARP refinement if still enabled after all checks
        if use_sharp and self.sharp_refiner is not None:
            img_float = image_tensor.float() / 255.0
            refined_attrs = self.sharp_refiner.refine(img_float, depth)

        # Convert to Gaussians
        if refined_attrs:
            from .gaussian_converter import equirect_to_gaussians_refined
            gaussians = equirect_to_gaussians_refined(
                image=image_tensor,
                depth=depth,
                grid=grid,
                refined_attrs=refined_attrs,
                scale_factor=scale_factor,
                thickness_ratio=thickness_ratio,
                depth_min=depth_min,
                depth_max=depth_max,
                validity_mask=validity_mask,
                scale_blend=kwargs.get('scale_blend', 0.8),
                opacity_blend=kwargs.get('opacity_blend', 1.0),
                color_blend=kwargs.get('color_blend', 0.5),
            )
        else:
            gaussians = equirect_to_gaussians(
                image=image_tensor,
                depth=depth,
                grid=grid,
                scale_factor=scale_factor,
                thickness_ratio=thickness_ratio,
                depth_min=depth_min,
                depth_max=depth_max,
                validity_mask=validity_mask,
                depth_gradient_threshold=depth_gradient_threshold,
            )
        
        # Generate sky dome if enabled
        if kwargs.get('sky_dome', True) and sky_threshold > 0:
            from .gaussian_converter import generate_sky_dome
            sky_gaussians = generate_sky_dome(
                image=image_tensor,
                depth=depth,
                grid=grid,
                sky_threshold=sky_threshold,
            )
            if sky_gaussians['means'].shape[0] > 0:
                for key in gaussians:
                    gaussians[key] = torch.cat(
                        [gaussians[key], sky_gaussians[key]], dim=0
                    )
        
        # Save output
        output_path = Path(output_path)
        if output_format == "splat":
            save_splat(gaussians, str(output_path))
        else:
            save_ply_gsplat(gaussians, str(output_path), sh_degree=sh_degree)
        
        # Compute result stats
        elapsed = time.time() - start_time
        file_size = output_path.stat().st_size
        splat_count = gaussians['means'].shape[0]
        
        # Compute actual depth range from valid points
        distances = gaussians['means'].norm(dim=-1)
        if distances.numel() > 0:
            depth_range = (distances.min().item(), distances.max().item())
        else:
            depth_range = (0.0, 0.0)
        
        return ConversionResult(
            output_path=str(output_path),
            splat_count=splat_count,
            file_size=file_size,
            processing_time=elapsed,
            depth_range=depth_range
        )
    
    def convert_ply_to_splat(self, ply_path: str, splat_path: str) -> None:
        """
        Convert existing PLY to compressed SPLAT format.
        
        Args:
            ply_path: Input PLY file path
            splat_path: Output SPLAT file path
        """
        from .splat_writer import convert_ply_to_splat
        convert_ply_to_splat(ply_path, splat_path)
    
    def clear_cache(self) -> None:
        """Clear cached spherical grids to free memory."""
        self._grid_cache.clear()

    @staticmethod
    def merge_plys(
        ply_paths: List[Union[str, Path]],
        transforms: List[np.ndarray],
        output_path: Union[str, Path],
        sh_degree: int = 0,
        reference_index: int = 0,
    ) -> dict:
        """
        Merge multiple 3DGS PLY files into a single PLY.

        Each PLY is transformed into a common world coordinate frame before
        concatenation. Use this when combining reconstructions from different
        camera positions or merging tiled reconstructions.

        Args:
            ply_paths: List of paths to input PLY files
            transforms: List of 4x4 camera-to-world (or local-to-global) matrices.
                        Each transform should place the corresponding PLY into
                        the shared world frame.
            output_path: Path for the merged output PLY
            sh_degree: SH degree for output (0 = DC only)
            reference_index: Index of PLY to use as reference frame (default 0).
                             This PLY's transform is used as the global origin.

        Returns:
            Dict with merge statistics:
                - total_gaussians: Total number of Gaussians in merged PLY
                - per_ply_counts: List of Gaussian counts from each input
                - output_path: Path to merged file

        Example:
            # Transforms from COLMAP or your pose estimation
            T0 = np.eye(4)  # Reference PLY at origin
            T1 = np.array([...])  # Second PLY's world transform

            SPAG4D.merge_plys(
                ply_paths=["scene_a.ply", "scene_b.ply"],
                transforms=[T0, T1],
                output_path="merged.ply"
            )
        """
        from .ply_writer import load_ply_gaussians, save_ply_gsplat, _quat_multiply

        if len(ply_paths) != len(transforms):
            raise ValueError(
                f"Number of PLYs ({len(ply_paths)}) must match "
                f"number of transforms ({len(transforms)})"
            )

        if len(ply_paths) == 0:
            raise ValueError("At least one PLY file required")

        # Compute reference transform inverse
        T_ref = transforms[reference_index]
        T_ref_inv = np.linalg.inv(T_ref)

        all_means = []
        all_scales = []
        all_quats = []
        all_colors = []
        all_opacities = []
        per_ply_counts = []

        for i, (ply_path, T_local) in enumerate(zip(ply_paths, transforms)):
            # Load Gaussians
            g = load_ply_gaussians(str(ply_path))

            means = g['means'].numpy()
            scales = g['scales'].numpy()
            quats = g['quats'].numpy()  # XYZW order
            colors = g['colors'].numpy()
            opacities = g['opacities'].numpy()

            per_ply_counts.append(len(means))

            # Compute transform: local -> global (relative to reference)
            # T_global = T_ref_inv @ T_local
            T_global = T_ref_inv @ T_local

            R = T_global[:3, :3]
            t = T_global[:3, 3]

            # Check for scale in the rotation matrix
            scale_factors = np.linalg.norm(R, axis=0)
            has_scale = not np.allclose(scale_factors, 1.0, atol=1e-4)

            if has_scale:
                # Extract pure rotation and apply scale separately
                R_pure = R / scale_factors
                scales_transformed = scales * scale_factors
            else:
                R_pure = R
                scales_transformed = scales

            # Transform positions: x' = R @ x + t
            means_transformed = (R_pure @ means.T).T + t

            # Transform rotations: q' = q_R * q_gaussian
            q_R = _rotation_matrix_to_quat_xyzw(R_pure)
            quats_transformed = _quat_multiply(q_R, quats)

            # Normalize quaternions
            quats_transformed = quats_transformed / np.linalg.norm(
                quats_transformed, axis=-1, keepdims=True
            )

            all_means.append(means_transformed)
            all_scales.append(scales_transformed)
            all_quats.append(quats_transformed)
            all_colors.append(colors)
            all_opacities.append(opacities)

        # Concatenate all Gaussians
        merged = {
            'means': torch.from_numpy(np.concatenate(all_means, axis=0)).float(),
            'scales': torch.from_numpy(np.concatenate(all_scales, axis=0)).float(),
            'quats': torch.from_numpy(np.concatenate(all_quats, axis=0)).float(),
            'colors': torch.from_numpy(np.concatenate(all_colors, axis=0)).float(),
            'opacities': torch.from_numpy(np.concatenate(all_opacities, axis=0)).float(),
        }

        # Save merged PLY
        save_ply_gsplat(merged, str(output_path), sh_degree=sh_degree)

        total = merged['means'].shape[0]
        print(f"Merged {len(ply_paths)} PLYs → {total:,} Gaussians")

        return {
            'total_gaussians': total,
            'per_ply_counts': per_ply_counts,
            'output_path': str(output_path),
        }

