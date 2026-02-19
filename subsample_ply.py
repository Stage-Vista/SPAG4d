#!/usr/bin/env python3
"""Subsample gaussians from a PLY file to at most N gaussians.

Usage:
    python subsample_ply.py input.ply output.ply [--max 2000000] [--strategy random|opacity]
"""

import argparse
import numpy as np
from plyfile import PlyData, PlyElement


def subsample_ply(input_path: str, output_path: str, max_gaussians: int, strategy: str) -> None:
    print(f"Reading {input_path} ...")
    ply = PlyData.read(input_path)
    vertex = ply.elements[0]
    n = vertex.count
    print(f"  Total gaussians: {n:,}")

    if n <= max_gaussians:
        print(f"  Already <= {max_gaussians:,}, copying as-is.")
        ply.write(output_path)
        return

    if strategy == "opacity":
        # Keep the most opaque (most visible) gaussians
        opacities = vertex.data["opacity"].astype(np.float32)
        # Opacity is stored as logit(alpha), so higher = more visible
        indices = np.argpartition(opacities, -max_gaussians)[-max_gaussians:]
        indices = np.sort(indices)
        print(f"  Strategy: keep top {max_gaussians:,} by opacity")
    else:
        # Uniform random subsample
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(n, size=max_gaussians, replace=False)
        indices = np.sort(indices)
        print(f"  Strategy: random subsample (seed=42)")

    subsampled = vertex.data[indices]
    new_element = PlyElement.describe(subsampled, vertex.name)
    PlyData([new_element], text=ply.text).write(output_path)
    print(f"  Written {max_gaussians:,} gaussians to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Subsample gaussians in a PLY file.")
    parser.add_argument("input", help="Input PLY file")
    parser.add_argument("output", help="Output PLY file")
    parser.add_argument("--max", type=int, default=2_000_000, metavar="N",
                        help="Maximum number of gaussians to keep (default: 2000000)")
    parser.add_argument("--strategy", choices=["random", "opacity"], default="random",
                        help="Subsampling strategy: 'random' (default) or 'opacity' (keep most visible)")
    args = parser.parse_args()

    subsample_ply(args.input, args.output, args.max, args.strategy)


if __name__ == "__main__":
    main()