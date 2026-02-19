#!/usr/bin/env python3
"""
pipeline_pano_video.py
----------------------
Convert a Matrix-3D panoramic video + camera NPZ into a merged Gaussian splat PLY.

Pipeline:
  1. Extract frames from pano_video.mp4  (one image per camera pose)
  2. Run SPAG4D.convert() on each frame  → per-frame PLY
  3. Run SPAG4D.merge_plys() with the camera-to-world transforms from the NPZ
     → single merged PLY in the shared world frame

Usage:
  python pipeline_pano_video.py \\
      --video   /path/to/pano_video.mp4 \\
      --cameras /path/to/condition/cameras.npz \\
      --output  /path/to/output_dir

Optional flags (all have sensible defaults):
  --device          cuda | cpu | mps        (default: cuda)
  --stride          1 | 2 | 4 | 8          (default: 2, spatial downsample)
  --scale-factor    float                  (default: 1.5, Gaussian scale)
  --thickness-ratio float                  (default: 0.1)
  --global-scale    float                  (default: 1.0, depth multiplier)
  --depth-min       float                  (default: 0.1 m)
  --depth-max       float                  (default: 100.0 m)
  --sky-threshold            float  (default: 80.0 m, 0 to disable)
  --sh-degree                0 | 3  (default: 0)
  --depth-gradient-threshold float  (default: 0.5; 0=off, 0.3=aggressive)
  --frame-step      int                    (default: 1, use every Nth frame)
  --frames-dir      path                   (skip extraction; use existing frames)
  --depth-dir       path                   (pre-computed metric DAP depth EXRs;
                                            named NNNN.exr, 0-indexed by video frame.
                                            When provided, bypasses PanDA so splat scale
                                            matches the metric cameras.npz poses.)
  --skip-merge                             (stop after PLY generation)
  --use-mock-dap                           (use mock depth model for testing)
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_frames(video_path: Path, frames_dir: Path) -> list[Path]:
    """Extract all video frames to PNG files using ffmpeg."""
    frames_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vsync", "0",
        str(frames_dir / "frame_%04d.png"),
    ]
    print(f"[1/3] Extracting frames → {frames_dir}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg stderr:", result.stderr, file=sys.stderr)
        raise RuntimeError("ffmpeg frame extraction failed")

    frames = sorted(frames_dir.glob("frame_*.png"))
    print(f"      Extracted {len(frames)} frames")
    return frames


def load_cameras(npz_path: Path) -> np.ndarray:
    """Load camera-to-world matrices from cameras.npz.

    Returns array of shape (N, 4, 4).
    """
    data = np.load(str(npz_path))
    # The Matrix-3D pipeline stores transforms under 'arr_0'
    key = "arr_0" if "arr_0" in data else list(data.keys())[0]
    transforms = data[key].astype(np.float32)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(
            f"Expected (N, 4, 4) array in {npz_path}, got {transforms.shape}"
        )
    return transforms


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    video_path   = Path(args.video)
    cameras_path = Path(args.cameras)
    output_dir   = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)
    plys_dir   = output_dir / "plys"
    frames_dir = Path(args.frames_dir) if args.frames_dir else output_dir / "frames"
    merged_ply = output_dir / "merged.ply"

    # ── Step 1: Frame extraction ──────────────────────────────────────────
    if args.frames_dir:
        frames = sorted(Path(args.frames_dir).glob("frame_*.png"))
        if not frames:
            frames = sorted(Path(args.frames_dir).glob("frame_*.jpg"))
        print(f"[1/3] Using existing frames from {args.frames_dir} ({len(frames)} found)")
    else:
        frames = extract_frames(video_path, frames_dir)

    # ── Load camera transforms ────────────────────────────────────────────
    transforms_all = load_cameras(cameras_path)
    n_cams = len(transforms_all)

    if len(frames) != n_cams:
        print(
            f"WARNING: {len(frames)} frames vs {n_cams} camera poses. "
            "Will use min(frames, poses)."
        )

    # Apply frame-step subsampling
    step = max(1, args.frame_step)
    frames      = frames[::step]
    transforms  = transforms_all[::step]

    print(
        f"      Processing {len(frames)} frames "
        f"(step={step}, total poses={n_cams})"
    )

    # ── Step 2: Per-frame conversion ──────────────────────────────────────
    print(f"[2/3] Converting frames → PLYs in {plys_dir}")
    plys_dir.mkdir(parents=True, exist_ok=True)

    from spag4d import SPAG4D

    # When --depth-dir is given every frame gets a precomputed metric depth map
    # so the internal depth model is never called.  Use "mock" to skip loading
    # PanDA/DAP weights (saves ~2 GB VRAM and several seconds of init time).
    depth_model = "mock" if args.depth_dir else ("mock" if args.use_mock_dap else "panda")
    converter = SPAG4D(
        device=args.device,
        depth_model=depth_model,
    )

    # Build depth-dir lookup: map 0-indexed video frame number → EXR path
    depth_dir = Path(args.depth_dir) if args.depth_dir else None

    ply_paths: list[Path] = []
    for i, frame_path in enumerate(frames):
        ply_path = plys_dir / f"{frame_path.stem}.ply"

        # Skip if already done (resume-friendly)
        if ply_path.exists():
            print(f"  [{i+1}/{len(frames)}] {frame_path.name} → (cached) {ply_path.name}")
            ply_paths.append(ply_path)
            continue

        # Resolve pre-computed depth map for this frame (if depth-dir provided).
        # ffmpeg names frames frame_0001.png (1-indexed); DAP temporal saves
        # {frame_idx:04d}.exr (0-indexed), so subtract 1 to match.
        precomputed_depth = None
        if depth_dir is not None:
            try:
                frame_num = int(frame_path.stem.split("_")[-1])
                depth_exr = depth_dir / f"{frame_num - 1:04d}.exr"
                if depth_exr.exists():
                    precomputed_depth = depth_exr
                else:
                    print(f"\n  [depth-dir] {depth_exr.name} not found, falling back to PanDA")
            except (ValueError, IndexError):
                pass

        print(f"  [{i+1}/{len(frames)}] {frame_path.name} → {ply_path.name}", end="", flush=True)
        result = converter.convert(
            input_path=frame_path,
            output_path=ply_path,
            stride=args.stride,
            scale_factor=args.scale_factor,
            thickness_ratio=args.thickness_ratio,
            global_scale=args.global_scale,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            sky_threshold=args.sky_threshold,
            sh_degree=args.sh_degree,
            output_format="ply",
            depth_gradient_threshold=args.depth_gradient_threshold,
            precomputed_depth=precomputed_depth,
        )
        print(f"  {result.splat_count:,} splats  ({result.processing_time:.1f}s)")
        ply_paths.append(ply_path)

    # Release GPU memory before merge
    converter.clear_cache()
    del converter

    if args.skip_merge:
        print(f"\nDone. PLYs written to {plys_dir}")
        return

    # ── Step 3: Merge PLYs ────────────────────────────────────────────────
    print(f"[3/3] Merging {len(ply_paths)} PLYs → {merged_ply}")

    # Align transforms list to the (possibly subsampled) ply_paths list
    used_transforms = [transforms[i] for i in range(len(ply_paths))]

    # cameras.npz stores w2c (world-to-camera) matrices from Matrix-3D's
    # nvrender pipeline.  merge_plys() expects c2w (camera-to-world).
    #
    # Additionally, SPAG4D generates Gaussians in its own Y-up spherical
    # frame (rhat = [sinφ cosθ, cosφ, −sinφ sinθ]) while the Matrix-3D
    # world frame places the panorama center along +Z (after its internal
    # rot_matrix).  These two frames are related by the constant rotation:
    #
    #   R_conv = [[0, 0,-1],   (SPAG4D X → M3D −Z)
    #             [0,-1, 0],   (SPAG4D Y → M3D −Y)
    #             [-1, 0, 0]]  (SPAG4D −Z → M3D −X, i.e. forward maps to +Z)
    #
    # The corrected per-frame transform passed to merge_plys is:
    #   T_corrected_i = inv(w2c_i) @ R_conv_4x4 = c2w_i @ R_conv_4x4
    #
    # merge_plys then computes T_global_i = inv(T_corrected_0) @ T_corrected_i
    # = R_conv^T @ w2c_0 @ inv(w2c_i) @ R_conv, which is correct.
    R_conv = np.array([
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [-1,  0,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float32)

    corrected_transforms = [
        np.linalg.inv(T) @ R_conv   # c2w_i @ R_conv
        for T in used_transforms
    ]

    stats = SPAG4D.merge_plys(
        ply_paths=[str(p) for p in ply_paths],
        transforms=corrected_transforms,
        output_path=str(merged_ply),
        sh_degree=args.sh_degree,
        reference_index=0,
    )

    size_mb = merged_ply.stat().st_size / (1024 ** 2)
    print(f"\nDone.")
    print(f"  Merged PLY : {merged_ply}")
    print(f"  Gaussians  : {stats['total_gaussians']:,}")
    print(f"  File size  : {size_mb:.1f} MB")
    per = ", ".join(f"{c:,}" for c in stats["per_ply_counts"])
    print(f"  Per-frame  : {per}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert a panoramic video + camera NPZ to a merged 3DGS PLY.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--video",   required=True, help="Path to pano_video.mp4")
    p.add_argument("--cameras", required=True, help="Path to cameras.npz (arr_0: N×4×4 c2w matrices)")
    p.add_argument("--output",  required=True, help="Output directory for frames, PLYs, and merged PLY")

    # Conversion params
    p.add_argument("--device",          default="cuda",  help="Compute device (cuda/cpu/mps)")
    p.add_argument("--stride",          type=int,   default=2,   help="Spatial downsample stride")
    p.add_argument("--scale-factor",    type=float, default=1.5, help="Gaussian scale multiplier")
    p.add_argument("--thickness-ratio", type=float, default=0.1, help="Radial thickness ratio")
    p.add_argument("--global-scale",    type=float, default=1.0, help="Depth scale multiplier")
    p.add_argument("--depth-min",       type=float, default=0.1, help="Min valid depth (m)")
    p.add_argument("--depth-max",       type=float, default=100.0, help="Max valid depth (m)")
    p.add_argument("--sky-threshold",          type=float, default=80.0, help="Sky depth threshold (m), 0=off")
    p.add_argument("--sh-degree",              type=int,   default=0,   choices=[0, 3], help="SH degree")
    p.add_argument("--depth-gradient-threshold", type=float, default=0.5,
                   help="Relative depth-gradient cutoff for boundary floater removal "
                        "(|∇d|/d > threshold → rejected). 0=off, 0.3=aggressive, 0.7=mild")

    # Pipeline control
    p.add_argument("--frame-step",  type=int,  default=1,   help="Use every Nth frame (1=all)")
    p.add_argument("--frames-dir",  default=None,            help="Use pre-extracted frames dir")
    p.add_argument("--depth-dir",   default=None,
                   help="Directory of pre-computed metric DAP depth EXRs "
                        "(named NNNN.exr, 0-indexed by video frame). "
                        "Bypasses PanDA so splat scale matches cameras.npz metric poses.")
    p.add_argument("--skip-merge",  action="store_true",     help="Stop after per-frame PLY generation")
    p.add_argument("--use-mock-dap", action="store_true",    help="Use mock DAP (for testing)")

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_pipeline(args)
