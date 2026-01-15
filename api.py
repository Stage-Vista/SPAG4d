# api.py
"""
FastAPI backend for SPAG-4D web UI.
"""

import asyncio
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from spag4d import SPAG4D, ConversionResult


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
TEMP_DIR = Path("./temp_spag4d")
JOB_TTL_SECONDS = 30 * 60  # 30 minutes
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
GPU_SEMAPHORE_LIMIT = 1  # Only 1 concurrent GPU job


# ─────────────────────────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────────────────────────
processor: Optional[SPAG4D] = None
gpu_semaphore: Optional[asyncio.Semaphore] = None
jobs: dict = {}  # job_id -> JobInfo


class JobInfo:
    """Tracks a conversion job."""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = "queued"
        self.created_at = time.time()
        self.last_updated = time.time()  # Track last activity
        self.input_path: Optional[Path] = None
        self.output_ply_path: Optional[Path] = None
        self.output_splat_path: Optional[Path] = None
        self.preview_splat_path: Optional[Path] = None
        self.result: Optional[ConversionResult] = None
        self.error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────
# Lifecycle Management
# ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    global processor, gpu_semaphore
    
    # Startup
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try to initialize with real DAP, fall back to mock if unavailable
    try:
        processor = SPAG4D(device="cuda", use_mock_dap=False)
        print("Loaded real DAP model")
    except Exception as e:
        print(f"DAP model not available ({e}), using mock depth")
        processor = SPAG4D(device="cuda", use_mock_dap=True)
    
    gpu_semaphore = asyncio.Semaphore(GPU_SEMAPHORE_LIMIT)
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(cleanup_loop())
    
    yield
    
    # Shutdown
    cleanup_task.cancel()
    await run_cleanup()


async def cleanup_loop():
    """Periodic cleanup of expired jobs and temp files."""
    while True:
        await asyncio.sleep(60)  # Check every minute
        await run_cleanup()


async def run_cleanup():
    """Remove expired jobs and their files."""
    now = time.time()
    
    # Only cleanup completed or errored jobs that are older than TTL
    # Never cleanup queued or processing jobs based on time
    expired_jobs = [
        job_id for job_id, job in jobs.items()
        if job.status in ("complete", "error") 
        and now - job.last_updated > JOB_TTL_SECONDS
    ]
    
    for job_id in expired_jobs:
        job = jobs.pop(job_id, None)
        if job:
            # Delete associated files
            for path in [job.input_path, job.output_ply_path, 
                        job.output_splat_path, job.preview_splat_path]:
                if path and path.exists():
                    try:
                        path.unlink()
                    except Exception:
                        pass
    
    # Also clean orphaned files in temp dir
    for f in TEMP_DIR.glob("*"):
        try:
            if now - f.stat().st_mtime > JOB_TTL_SECONDS:
                f.unlink()
        except Exception:
            pass


app = FastAPI(title="SPAG-4D", lifespan=lifespan)


# ─────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────
@app.post("/api/convert")
async def convert_panorama(
    file: UploadFile = File(...),
    stride: int = Query(2, ge=1, le=8),
    scale_factor: float = Query(1.5, ge=0.1, le=5.0),
    thickness: float = Query(0.1, ge=0.01, le=1.0),
    global_scale: float = Query(1.0, ge=0.1, le=10.0),
    depth_min: float = Query(0.1, ge=0.01),
    depth_max: float = Query(100.0, le=1000.0)
):
    """
    Convert uploaded panorama to Gaussian splat.
    
    Returns job_id immediately. Poll /api/status/{job_id} for progress.
    """
    # Validate file size
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(400, f"File too large. Max: {MAX_UPLOAD_SIZE // 1024 // 1024}MB")
    
    # Create job
    job_id = str(uuid.uuid4())
    job = JobInfo(job_id)
    jobs[job_id] = job
    
    # Determine file extension
    suffix = Path(file.filename).suffix if file.filename else '.jpg'
    
    # Save input file
    job.input_path = TEMP_DIR / f"{job_id}_input{suffix}"
    job.output_ply_path = TEMP_DIR / f"{job_id}_output.ply"
    job.output_splat_path = TEMP_DIR / f"{job_id}_output.splat"
    job.preview_splat_path = TEMP_DIR / f"{job_id}_preview.splat"
    
    with open(job.input_path, "wb") as f:
        f.write(content)
    
    # Queue processing (non-blocking)
    asyncio.create_task(process_job(
        job, stride, scale_factor, thickness, global_scale, depth_min, depth_max
    ))
    
    return JSONResponse({
        "job_id": job_id,
        "status": "queued",
        "queue_position": sum(1 for j in jobs.values() if j.status == "queued")
    })


async def process_job(
    job: JobInfo,
    stride: int,
    scale_factor: float,
    thickness: float,
    global_scale: float,
    depth_min: float,
    depth_max: float
):
    """Process conversion job with GPU semaphore."""
    global processor
    
    try:
        # Wait for GPU access
        job.status = "queued"
        async with gpu_semaphore:
            job.status = "processing"
            
            # Run heavy computation in thread pool (doesn't block event loop)
            result = await run_in_threadpool(
                processor.convert,
                input_path=str(job.input_path),
                output_path=str(job.output_ply_path),
                stride=stride,
                scale_factor=scale_factor,
                thickness_ratio=thickness,
                global_scale=global_scale,
                depth_min=depth_min,
                depth_max=depth_max
            )
            
            # Generate web preview (low-res SPLAT)
            await run_in_threadpool(
                processor.convert,
                input_path=str(job.input_path),
                output_path=str(job.preview_splat_path),
                stride=8,  # Always use stride 8 for preview
                scale_factor=scale_factor,
                thickness_ratio=thickness,
                global_scale=global_scale,
                depth_min=depth_min,
                depth_max=depth_max,
                output_format="splat"
            )
            
            # Also generate full SPLAT for download
            await run_in_threadpool(
                processor.convert_ply_to_splat,
                str(job.output_ply_path),
                str(job.output_splat_path)
            )
            
            job.result = result
            job.status = "complete"
            job.last_updated = time.time()
            
            # Delete input file immediately after processing
            if job.input_path and job.input_path.exists():
                job.input_path.unlink()
                
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        job.last_updated = time.time()


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and result."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    
    job = jobs[job_id]
    
    response = {
        "job_id": job_id,
        "status": job.status,
    }
    
    if job.status == "queued":
        response["queue_position"] = sum(
            1 for j in jobs.values() 
            if j.status == "queued" and j.created_at < job.created_at
        ) + 1
    
    if job.status == "complete" and job.result:
        response["splat_count"] = job.result.splat_count
        response["file_size_mb"] = round(job.result.file_size / 1024 / 1024, 2)
        response["processing_time"] = round(job.result.processing_time, 2)
        response["preview_url"] = f"/api/preview/{job_id}"
    
    if job.status == "error":
        response["error"] = job.error
    
    return JSONResponse(response)


@app.get("/api/preview/{job_id}")
async def get_preview(job_id: str):
    """Get low-res SPLAT preview for web viewer."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    
    job = jobs[job_id]
    if job.status != "complete":
        raise HTTPException(400, "Job not complete")
    
    if not job.preview_splat_path or not job.preview_splat_path.exists():
        raise HTTPException(404, "Preview not available")
    
    return FileResponse(
        job.preview_splat_path,
        media_type="application/octet-stream",
        filename=f"preview_{job_id[:8]}.splat"
    )


@app.get("/api/download/{job_id}")
async def download_file(job_id: str, format: str = "ply"):
    """Download the generated file."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    
    job = jobs[job_id]
    if job.status != "complete":
        raise HTTPException(400, "Job not complete")
    
    if format == "splat":
        path = job.output_splat_path
        filename = f"spag4d_{job_id[:8]}.splat"
    else:
        path = job.output_ply_path
        filename = f"spag4d_{job_id[:8]}.ply"
    
    if not path or not path.exists():
        raise HTTPException(404, "File not found")
    
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=filename
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "gpu_available": gpu_semaphore._value > 0 if gpu_semaphore else False,
        "active_jobs": sum(1 for j in jobs.values() if j.status == "processing"),
        "queued_jobs": sum(1 for j in jobs.values() if j.status == "queued"),
    }


# Serve test images
TEST_IMAGE_DIR = Path("./TestImage")
if TEST_IMAGE_DIR.exists():
    app.mount("/TestImage", StaticFiles(directory="TestImage"), name="test-images")

# Serve static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")
