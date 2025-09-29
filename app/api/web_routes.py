from fastapi import APIRouter, Request, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import json
import tempfile
import os
import uuid
from datetime import datetime
import shutil
from typing import Optional

from app.models import ReframeRequest, ReframeSettings, GeminiPrompts, JobStatus
from app.services.s3_service import S3Service

web_router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

def get_redis():
    """Dependency to get Redis client"""
    from app.main import redis_client as rc
    return rc

def get_video_processor():
    """Dependency to get video processor"""
    from app.main import video_processor as vp
    return vp

@web_router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@web_router.post("/api/v1/upload-and-reframe")
async def upload_and_reframe(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    settings: str = Form(...),
    redis=Depends(get_redis),
    processor=Depends(get_video_processor)
):
    """Handle video upload and start reframing process"""

    # Validate file
    if not video.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Check file size (500MB limit)
    max_size = 500 * 1024 * 1024  # 500MB
    file_size = 0

    # Create temporary file to save upload
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

    try:
        # Save uploaded file to temp location
        while True:
            chunk = await video.read(8192)  # Read in 8KB chunks
            if not chunk:
                break

            file_size += len(chunk)
            if file_size > max_size:
                temp_file.close()
                os.unlink(temp_file.name)
                raise HTTPException(status_code=400, detail="File size exceeds 500MB limit")

            temp_file.write(chunk)

        temp_file.close()

        # Upload to S3
        s3_service = S3Service()
        job_id = str(uuid.uuid4())
        s3_key = f"uploads/{job_id}/{video.filename}"

        input_url = await s3_service.upload_file(temp_file.name, s3_key)

        # Parse settings
        try:
            settings_data = json.loads(settings)
            reframe_settings = ReframeSettings(**settings_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid settings: {e}")

        # Create reframe request - put directly in reframe folder with unique ID in filename
        # Extract base filename without extension
        filename_base = video.filename.rsplit('.', 1)[0] if '.' in video.filename else video.filename
        output_key = f"reframe/{job_id}_{filename_base}_reframed.mp4"

        reframe_request = ReframeRequest(
            input_url=input_url,
            output_key=output_key,
            settings=reframe_settings,
            gemini_prompts=GeminiPrompts(),
            preview_only=False
        )

        # Initialize job in Redis
        job_data = {
            'job_id': job_id,
            'status': JobStatus.QUEUED.value,
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
            'progress': 0.0,
            'message': 'Job queued for processing',
            'request': json.dumps(reframe_request.dict()),
            'original_filename': video.filename
        }

        await redis.hset(f"job:{job_id}", mapping=job_data)
        await redis.expire(f"job:{job_id}", 86400)  # Expire after 24 hours

        # Start processing
        background_tasks.add_task(
            processor.process_video,
            job_id,
            reframe_request
        )

        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Video uploaded successfully, processing started"
        }

    except Exception as e:
        # Cleanup temp file on error
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Always try to cleanup temp file
        try:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        except:
            pass

@web_router.get("/api/v1/download/{job_id}")
async def download_result(job_id: str, redis=Depends(get_redis)):
    """Download the processed video"""

    # Get job data
    job_data = await redis.hgetall(f"job:{job_id}")

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    if job_data.get('status') != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Job not completed")

    output_url = job_data.get('output_url')
    if not output_url:
        raise HTTPException(status_code=404, detail="Output file not found")

    # For S3 URLs, redirect to the URL (files are public)
    if output_url.startswith('https://'):
        return {"download_url": output_url}

    raise HTTPException(status_code=404, detail="File not accessible")

@web_router.get("/api/v1/preview/{job_id}")
async def get_preview(job_id: str, redis=Depends(get_redis)):
    """Get preview video URL"""

    job_data = await redis.hgetall(f"job:{job_id}")

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    preview_url = job_data.get('preview_url')
    if not preview_url:
        raise HTTPException(status_code=404, detail="Preview not available")

    return {"preview_url": preview_url}

@web_router.get("/api/v1/gallery")
async def get_recent_videos(limit: int = 20, redis=Depends(get_redis)):
    """Get recent processed videos for gallery view"""

    # Get all completed jobs
    job_keys = await redis.keys("job:*")
    completed_jobs = []

    for key in job_keys:
        job_data = await redis.hgetall(key)
        if job_data and job_data.get('status') == JobStatus.COMPLETED.value:
            completed_jobs.append({
                'job_id': job_data['job_id'],
                'filename': job_data.get('original_filename', 'Unknown'),
                'created_at': job_data['created_at'],
                'output_url': job_data.get('output_url'),
                'preview_url': job_data.get('preview_url'),
                'analytics': json.loads(job_data['analytics']) if job_data.get('analytics') else None
            })

    # Sort by creation time (newest first)
    completed_jobs.sort(key=lambda x: x['created_at'], reverse=True)

    return {
        'videos': completed_jobs[:limit],
        'total': len(completed_jobs)
    }

@web_router.get("/gallery", response_class=HTMLResponse)
async def gallery_page(request: Request):
    """Serve the gallery page"""
    return templates.TemplateResponse("gallery.html", {"request": request})

@web_router.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    """Serve the about page"""
    return templates.TemplateResponse("about.html", {"request": request})

@web_router.get("/help", response_class=HTMLResponse)
async def help_page(request: Request):
    """Serve the help page"""
    return templates.TemplateResponse("help.html", {"request": request})