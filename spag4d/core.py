# spag4d/core.py
"""
Main SPAG4D class that orchestrates the conversion pipeline.
"""

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Literal, Union
import time

from PIL import Image, ImageOps


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
    
    Uses DAP for native equirectangular depth estimation
    and SPAG algorithm for Gaussian splat generation.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[str] = None,
        use_mock_dap: bool = False
    ):
        """
        Initialize SPAG4D converter.
        
        Args:
            device: Device for computation ("cuda", "cpu", "mps")
            model_path: Optional explicit path to DAP weights
            use_mock_dap: Use mock DAP model (for testing without weights)
        """
        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        
        # Load DAP model
        if use_mock_dap:
            from .dap_model import MockDAPModel
            self.dap = MockDAPModel.load(device=self.device)
        else:
            from .dap_model import DAPModel
            self.dap = DAPModel.load(model_path, device=self.device)
        
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
        force_erp: bool = False
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
        
        # Estimate depth with DAP
        with torch.inference_mode():
            depth = self.dap.predict(image_tensor)
        
        # Apply global scale correction
        depth = depth * global_scale
        
        # Apply sky threshold
        if sky_threshold > 0:
            sky_mask = depth > sky_threshold
            depth[sky_mask] = 0  # Will be filtered by validity mask
        
        # Convert to Gaussians
        gaussians = equirect_to_gaussians(
            image=image_tensor,
            depth=depth,
            grid=grid,
            scale_factor=scale_factor,
            thickness_ratio=thickness_ratio,
            depth_min=depth_min,
            depth_max=depth_max
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
