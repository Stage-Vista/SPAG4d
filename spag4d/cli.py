# spag4d/cli.py
"""
Command-line interface for SPAG-4D.
"""

import click
from pathlib import Path
from typing import Optional


@click.group()
@click.version_option(version="0.2.0")
def main():
    """SPAG-4D: Convert 360° panoramas to 3D Gaussian Splats."""
    pass


@main.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--stride', default=2, help='Downsampling factor: 1, 2, 4, 8')
@click.option('--scale-factor', default=1.5, help='Gaussian scale multiplier')
@click.option('--thickness', default=0.1, help='Radial thickness ratio')
@click.option('--global-scale', default=1.0, help='Depth scale multiplier')
@click.option('--depth-min', default=0.1, help='Minimum depth in meters')
@click.option('--depth-max', default=100.0, help='Maximum depth in meters')
@click.option('--sky-threshold', default=80.0, help='Sky depth threshold (0 to disable)')
@click.option('--format', 'output_format', default='ply', 
              type=click.Choice(['ply', 'splat', 'both']),
              help='Output format')
@click.option('--sh-degree', default='0', type=click.Choice(['0', '3']),
              help='SH degree (0 or 3)')
@click.option('--force-erp', is_flag=True, help='Process even if aspect ratio isn\'t 2:1')
@click.option('--batch', is_flag=True, help='Process all images in input directory')
@click.option('--device', default='cuda', help='Device: cuda, cpu, mps')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
@click.option('--mock-dap', is_flag=True, help='Use mock DAP model (for testing)')
def convert(
    input_path: str,
    output_path: str,
    stride: int,
    scale_factor: float,
    thickness: float,
    global_scale: float,
    depth_min: float,
    depth_max: float,
    sky_threshold: float,
    output_format: str,
    sh_degree: int,
    force_erp: bool,
    batch: bool,
    device: str,
    quiet: bool,
    mock_dap: bool
):
    """
    Convert equirectangular panorama to Gaussian splat.
    
    INPUT_PATH: Input ERP image or directory
    OUTPUT_PATH: Output PLY/SPLAT file or directory
    """
    from .core import SPAG4D
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Initialize converter
    if not quiet:
        click.echo("Loading SPAG-4D...")
    
    converter = SPAG4D(device=device, use_mock_dap=mock_dap)
    
    if batch:
        # Batch mode: process all images in directory
        if not input_path.is_dir():
            raise click.ClickException("Input path must be a directory for batch mode")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.tiff'}
        images = [f for f in input_path.iterdir() if f.suffix.lower() in image_exts]
        
        if not quiet:
            click.echo(f"Processing {len(images)} images...")
        
        for img_path in images:
            out_name = img_path.stem + ('.splat' if output_format == 'splat' else '.ply')
            out_path = output_path / out_name
            
            try:
                result = converter.convert(
                    input_path=str(img_path),
                    output_path=str(out_path),
                    stride=stride,
                    scale_factor=scale_factor,
                    thickness_ratio=thickness,
                    global_scale=global_scale,
                    depth_min=depth_min,
                    depth_max=depth_max,
                    sky_threshold=sky_threshold,
                    sh_degree=int(sh_degree),
                    output_format='splat' if output_format == 'splat' else 'ply',
                    force_erp=force_erp
                )
                
                if not quiet:
                    click.echo(f"  ✓ {img_path.name} → {result.splat_count:,} splats")
                
                # Also generate SPLAT if format is 'both'
                if output_format == 'both':
                    splat_path = output_path / (img_path.stem + '.splat')
                    converter.convert_ply_to_splat(str(out_path), str(splat_path))
                    
            except Exception as e:
                click.echo(f"  ✗ {img_path.name}: {e}", err=True)
    
    else:
        # Single file mode
        fmt = 'splat' if output_format == 'splat' else 'ply'
        
        result = converter.convert(
            input_path=str(input_path),
            output_path=str(output_path),
            stride=stride,
            scale_factor=scale_factor,
            thickness_ratio=thickness,
            global_scale=global_scale,
            depth_min=depth_min,
            depth_max=depth_max,
            sky_threshold=sky_threshold,
            sh_degree=int(sh_degree),
            output_format=fmt,
            force_erp=force_erp
        )
        
        if not quiet:
            click.echo(f"Converted: {result.splat_count:,} Gaussians")
            click.echo(f"File size: {result.file_size / 1024 / 1024:.2f} MB")
            click.echo(f"Time: {result.processing_time:.2f}s")
            click.echo(f"Depth range: {result.depth_range[0]:.2f}m - {result.depth_range[1]:.2f}m")
        
        # Also generate SPLAT if format is 'both'
        if output_format == 'both':
            splat_path = Path(output_path).with_suffix('.splat')
            converter.convert_ply_to_splat(str(output_path), str(splat_path))
            if not quiet:
                click.echo(f"Also saved: {splat_path}")


@main.command('download-models')
@click.option('--verify', is_flag=True, help='Verify downloaded weights')
def download_models(verify: bool):
    """Download and cache DAP model weights."""
    from .dap_model import DAPModel, DAP_CACHE_DIR
    
    click.echo("Downloading DAP model weights...")
    
    try:
        path = DAPModel._get_or_download_weights()
        click.echo(f"✓ Weights cached at: {path}")
        
        if verify:
            if DAPModel._verify_checksum(Path(path)):
                click.echo("✓ Checksum verified")
            else:
                click.echo("⚠ Checksum verification skipped (no reference hash)")
    except Exception as e:
        click.echo(f"✗ Download failed: {e}", err=True)
        raise click.Abort()


@main.command()
@click.option('--port', default=7860, help='Server port')
@click.option('--host', default='127.0.0.1', help='Server host')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def serve(port: int, host: str, reload: bool):
    """Start the web UI server."""
    try:
        import uvicorn
    except ImportError:
        raise click.ClickException(
            "uvicorn not installed. Install with: pip install uvicorn"
        )
    
    click.echo(f"Starting SPAG-4D web UI at http://{host}:{port}")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload
    )


@main.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--fps', default=10, help='Frames per second to extract')
@click.option('--stride', default=4, help='Downsampling factor')
@click.option('--device', default='cuda', help='Device')
def video(input_video: str, output_dir: str, fps: int, stride: int, device: str):
    """
    Extract frames from 360° video and convert each to Gaussian splat.
    
    ⚠️ Warning: Frame-by-frame processing will have temporal flickering.
    """
    import subprocess
    import tempfile
    from .core import SPAG4D
    
    input_video = Path(input_video)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frames to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        click.echo(f"Extracting frames at {fps} FPS...")
        
        subprocess.run([
            'ffmpeg', '-i', str(input_video),
            '-vf', f'fps={fps}',
            '-qscale:v', '2',
            str(tmpdir / 'frame_%05d.jpg')
        ], check=True, capture_output=True)
        
        frames = sorted(tmpdir.glob('frame_*.jpg'))
        click.echo(f"Extracted {len(frames)} frames")
        
        # Convert each frame
        converter = SPAG4D(device=device)
        
        with click.progressbar(frames, label='Converting frames') as bar:
            for frame_path in bar:
                out_path = output_dir / (frame_path.stem + '.ply')
                try:
                    converter.convert(
                        input_path=str(frame_path),
                        output_path=str(out_path),
                        stride=stride,
                        force_erp=True
                    )
                except Exception as e:
                    click.echo(f"\n⚠ {frame_path.name}: {e}", err=True)
    
    click.echo(f"✓ Converted {len(frames)} frames to {output_dir}")


if __name__ == '__main__':
    main()
