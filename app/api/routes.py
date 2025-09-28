from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends
from typing import Optional
import json
import asyncio
import logging
import uuid
from datetime import datetime
import httpx

from app.models import ReframeRequest, JobResponse, JobStatus
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Global references (will be set in main.py)
redis_client = None
video_processor = None

def get_redis():
    """Dependency to get Redis client"""
    from app.main import redis_client as rc
    return rc

def get_video_processor():
    """Dependency to get video processor"""
    from app.main import video_processor as vp
    return vp

@router.post("/reframe", response_model=JobResponse)
async def create_reframe_job(
    request: ReframeRequest,
    background_tasks: BackgroundTasks,
    redis=Depends(get_redis),
    processor=Depends(get_video_processor)
):
    """Create a new video reframing job"""

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Validate input URL
    if not await validate_input_url(request.input_url):
        raise HTTPException(status_code=400, detail="Invalid or inaccessible input URL")

    # Pre-validate Gemini AI health before starting processing
    try:
        gemini_health = await processor.gemini.health_check()

        if gemini_health["status"] == "model_not_found":
            raise HTTPException(
                status_code=503,
                detail=f"AI service unavailable: {gemini_health.get('error', 'Model not found')}"
            )
        elif gemini_health["status"] == "connection_failed":
            raise HTTPException(
                status_code=503,
                detail="AI service connection failed. Please try again later."
            )
        elif gemini_health["status"] == "quota_exceeded":
            raise HTTPException(
                status_code=429,
                detail="AI service quota exceeded. Please try again later."
            )
        elif gemini_health["status"] not in ["healthy", "degraded"]:
            logger.warning(f"Gemini health check returned: {gemini_health}")
            # Allow processing to continue for other statuses but log warning

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Gemini health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="AI service health check failed. Please try again later."
        )

    # Initialize job in Redis
    job_data = {
        'job_id': job_id,
        'status': JobStatus.QUEUED.value,
        'created_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat(),
        'progress': 0.0,
        'message': 'Job queued for processing',
        'request': json.dumps(request.dict())
    }

    await redis.hset(f"job:{job_id}", mapping=job_data)
    await redis.expire(f"job:{job_id}", 86400)  # Expire after 24 hours

    # Add to processing queue
    background_tasks.add_task(
        processor.process_video,
        job_id,
        request
    )

    logger.info(f"Created reframing job {job_id}")

    return JobResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        progress=0.0,
        message='Job queued for processing'
    )

@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str, redis=Depends(get_redis)):
    """Get status of a reframing job"""

    job_data = await redis.hgetall(f"job:{job_id}")

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    # Parse analytics if present
    analytics = None
    if job_data.get('analytics'):
        try:
            analytics = json.loads(job_data['analytics'])
        except json.JSONDecodeError:
            analytics = None

    return JobResponse(
        job_id=job_data['job_id'],
        status=JobStatus(job_data['status']),
        created_at=datetime.fromisoformat(job_data['created_at']),
        updated_at=datetime.fromisoformat(job_data['updated_at']),
        progress=float(job_data.get('progress', 0)),
        message=job_data.get('message'),
        output_url=job_data.get('output_url'),
        preview_url=job_data.get('preview_url'),
        analytics=analytics,
        error=job_data.get('error')
    )

@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, redis=Depends(get_redis)):
    """Cancel a running job"""

    job_data = await redis.hgetall(f"job:{job_id}")

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    current_status = job_data.get('status')
    if current_status in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
        raise HTTPException(status_code=400, detail="Job already completed")

    # Update status to cancelled
    await redis.hset(f"job:{job_id}", mapping={
        'status': JobStatus.CANCELLED.value,
        'updated_at': datetime.utcnow().isoformat(),
        'message': 'Job cancelled by user'
    })

    logger.info(f"Cancelled job {job_id}")

    return {"message": "Job cancelled successfully", "job_id": job_id}

@router.get("/jobs")
async def list_jobs(
    limit: int = 50,
    offset: int = 0,
    status: Optional[JobStatus] = None,
    redis=Depends(get_redis)
):
    """List recent jobs with optional filtering"""

    # Get all job keys
    pattern = "job:*"
    job_keys = await redis.keys(pattern)

    # Sort by creation time (newest first)
    jobs = []
    for key in job_keys:
        job_data = await redis.hgetall(key)
        if job_data:
            # Filter by status if specified
            if status and job_data.get('status') != status.value:
                continue

            jobs.append({
                'job_id': job_data['job_id'],
                'status': job_data['status'],
                'created_at': job_data['created_at'],
                'progress': float(job_data.get('progress', 0)),
                'message': job_data.get('message', '')
            })

    # Sort by creation time
    jobs.sort(key=lambda x: x['created_at'], reverse=True)

    # Apply pagination
    total = len(jobs)
    jobs = jobs[offset:offset + limit]

    return {
        'jobs': jobs,
        'total': total,
        'limit': limit,
        'offset': offset
    }

@router.post("/jobs/{job_id}/retry")
async def retry_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    redis=Depends(get_redis),
    processor=Depends(get_video_processor)
):
    """Retry a failed job"""

    job_data = await redis.hgetall(f"job:{job_id}")

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    if job_data.get('status') != JobStatus.FAILED.value:
        raise HTTPException(status_code=400, detail="Job is not in failed state")

    # Parse original request
    try:
        request_data = json.loads(job_data['request'])
        request = ReframeRequest(**request_data)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid job request data: {e}")

    # Reset job status
    await redis.hset(f"job:{job_id}", mapping={
        'status': JobStatus.QUEUED.value,
        'progress': 0.0,
        'message': 'Job retry queued',
        'updated_at': datetime.utcnow().isoformat(),
        'error': '',
        'output_url': '',
        'preview_url': '',
        'analytics': ''
    })

    # Requeue for processing
    background_tasks.add_task(
        processor.process_video,
        job_id,
        request
    )

    logger.info(f"Retrying job {job_id}")

    return {"message": "Job retry initiated", "job_id": job_id}

@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job progress"""

    await websocket.accept()

    try:
        while True:
            # Get Redis client
            from app.main import redis_client

            # Get job status from Redis
            job_data = await redis_client.hgetall(f"job:{job_id}")

            if not job_data:
                await websocket.send_json({"error": "Job not found"})
                break

            # Send status update
            update = {
                'job_id': job_id,
                'status': job_data['status'],
                'progress': float(job_data.get('progress', 0)),
                'message': job_data.get('message', ''),
                'updated_at': job_data['updated_at']
            }

            await websocket.send_json(update)

            # Check if job is complete
            if job_data['status'] in [
                JobStatus.COMPLETED.value,
                JobStatus.FAILED.value,
                JobStatus.CANCELLED.value
            ]:
                # Send final update with results
                final_update = update.copy()
                final_update.update({
                    'output_url': job_data.get('output_url'),
                    'preview_url': job_data.get('preview_url'),
                    'error': job_data.get('error')
                })

                # Include analytics if available
                if job_data.get('analytics'):
                    try:
                        final_update['analytics'] = json.loads(job_data['analytics'])
                    except json.JSONDecodeError:
                        pass

                await websocket.send_json(final_update)
                break

            # Wait before next update
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {str(e)}")
        try:
            await websocket.send_json({"error": f"Internal error: {str(e)}"})
        except:
            pass
        await websocket.close()

@router.get("/health/gemini")
async def gemini_health_check(processor=Depends(get_video_processor)):
    """Check Gemini AI service health"""

    try:
        health_result = await processor.gemini.health_check()

        # Set HTTP status code based on health
        status_code = 200
        if health_result['status'] in ['connection_failed', 'model_not_found']:
            status_code = 503  # Service Unavailable
        elif health_result['status'] in ['degraded', 'quota_exceeded', 'safety_blocked']:
            status_code = 200  # OK but with warnings

        return health_result
    except Exception as e:
        return {
            'status': 'critical_error',
            'error': str(e),
            'response_time_ms': 0,
            'model': settings.GEMINI_MODEL,
            'last_checked': None
        }

@router.get("/health")
async def health_check(processor=Depends(get_video_processor)):
    """Health check endpoint"""

    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

    # Check Redis connection
    try:
        from app.main import redis_client
        await redis_client.ping()
        health_status["redis"] = "connected"
    except Exception as e:
        health_status["redis"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    # Check S3 connectivity
    try:
        from app.services.s3_service import S3Service
        s3 = S3Service()
        await s3.list_files(prefix="health-check", bucket=settings.S3_BUCKET)
        health_status["s3"] = "connected"
    except Exception as e:
        health_status["s3"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    # Check Gemini AI health
    try:
        gemini_health = await processor.gemini.health_check()
        health_status["gemini"] = {
            "status": gemini_health["status"],
            "model": gemini_health["model"],
            "response_time_ms": gemini_health["response_time_ms"]
        }

        # Update overall status if Gemini is unhealthy
        if gemini_health["status"] in ["connection_failed", "model_not_found"]:
            health_status["status"] = "unhealthy"
        elif gemini_health["status"] in ["degraded", "quota_exceeded", "safety_blocked"]:
            if health_status["status"] == "healthy":
                health_status["status"] = "degraded"

    except Exception as e:
        health_status["gemini"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    return health_status

@router.get("/metrics")
async def get_metrics(redis=Depends(get_redis)):
    """Get service metrics"""

    # Get job statistics
    job_keys = await redis.keys("job:*")
    job_stats = {
        'total': 0,
        'queued': 0,
        'processing': 0,
        'completed': 0,
        'failed': 0,
        'cancelled': 0
    }

    for key in job_keys:
        job_data = await redis.hgetall(key)
        if job_data:
            status = job_data.get('status', 'unknown')
            job_stats['total'] += 1
            job_stats[status] = job_stats.get(status, 0) + 1

    # Calculate processing statistics
    processing_time_total = 0
    completed_jobs = 0

    for key in job_keys:
        job_data = await redis.hgetall(key)
        if job_data and job_data.get('status') == JobStatus.COMPLETED.value:
            try:
                created = datetime.fromisoformat(job_data['created_at'])
                updated = datetime.fromisoformat(job_data['updated_at'])
                processing_time = (updated - created).total_seconds()
                processing_time_total += processing_time
                completed_jobs += 1
            except (ValueError, KeyError):
                pass

    avg_processing_time = processing_time_total / completed_jobs if completed_jobs > 0 else 0

    return {
        "job_statistics": job_stats,
        "average_processing_time": avg_processing_time,
        "active_connections": 0,  # Would track WebSocket connections
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/test")
async def test_endpoint(
    background_tasks: BackgroundTasks,
    redis=Depends(get_redis)
):
    """Test endpoint for development"""

    test_request = ReframeRequest(
        input_url="s3://test-bucket/test-video.mp4",
        settings={
            "mode": "auto",
            "quality": "medium"
        },
        preview_only=True
    )

    # Create test job
    job_id = str(uuid.uuid4())
    job_data = {
        'job_id': job_id,
        'status': JobStatus.QUEUED.value,
        'created_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat(),
        'progress': 0.0,
        'message': 'Test job created'
    }

    await redis.hset(f"job:{job_id}", mapping=job_data)

    return {
        "message": "Test job created",
        "job_id": job_id,
        "request": test_request.dict()
    }

async def validate_input_url(url: str) -> bool:
    """Validate that input URL is accessible"""

    try:
        if url.startswith('s3://'):
            # For S3 URLs, we'll validate during processing
            return True
        elif url.startswith('https://'):
            # For HTTPS URLs, try a HEAD request
            async with httpx.AsyncClient() as client:
                response = await client.head(url, timeout=10)
                return response.status_code == 200
        else:
            return False
    except Exception as e:
        logger.warning(f"URL validation failed for {url}: {e}")
        return False

@router.get("/")
async def root():
    """Root endpoint with API information"""

    return {
        "service": "Intelligent Video Reframer",
        "version": "1.0.0",
        "description": "AI-powered video reframing from 16:9 to 9:16 using Gemini",
        "endpoints": {
            "POST /api/v1/reframe": "Create reframing job",
            "GET /api/v1/jobs/{job_id}": "Get job status",
            "DELETE /api/v1/jobs/{job_id}": "Cancel job",
            "GET /api/v1/jobs": "List jobs",
            "WS /api/v1/ws/{job_id}": "Real-time job updates",
            "GET /api/v1/health": "Health check",
            "GET /api/v1/metrics": "Service metrics"
        },
        "documentation": "/docs"
    }