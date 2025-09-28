from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import redis.asyncio as redis
from app.config import settings
from app.models import ReframeRequest, JobStatus, JobResponse
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
redis_client = None
video_processor = None

async def perform_startup_health_checks(video_processor):
    """Perform comprehensive health checks on startup"""

    logger.info("Performing startup health checks...")

    # Check S3 connectivity
    try:
        from app.services.s3_service import S3Service
        s3 = S3Service()
        await s3.list_files(prefix="health-check", bucket=settings.S3_BUCKET)
        logger.info("✓ S3 connection: OK")
    except Exception as e:
        logger.warning(f"⚠ S3 connection: {e}")

    # Check Gemini AI service
    try:
        health_result = await video_processor.gemini.health_check()

        if health_result['status'] == 'healthy':
            logger.info(f"✓ Gemini AI: {health_result['status']} ({health_result['response_time_ms']}ms)")
        elif health_result['status'] in ['degraded', 'safety_blocked']:
            logger.warning(f"⚠ Gemini AI: {health_result['status']} ({health_result['response_time_ms']}ms)")
        else:
            logger.error(f"✗ Gemini AI: {health_result['status']} - {health_result.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"✗ Gemini AI health check failed: {e}")

    # Check FFmpeg availability
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info("✓ FFmpeg: Available")
        else:
            logger.warning("⚠ FFmpeg: Command failed")
    except subprocess.TimeoutExpired:
        logger.warning("⚠ FFmpeg: Timeout")
    except FileNotFoundError:
        logger.error("✗ FFmpeg: Not found in PATH")
    except Exception as e:
        logger.warning(f"⚠ FFmpeg check failed: {e}")

    logger.info("Startup health checks completed")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global redis_client, video_processor

    try:
        redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("Connected to Redis")

        # Import here to avoid circular imports
        from app.core.video_processor import VideoProcessor
        video_processor = VideoProcessor(redis_client)
        logger.info("Video processor initialized")

        # Perform startup health checks
        await perform_startup_health_checks(video_processor)

    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

    yield

    # Shutdown
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")

app = FastAPI(
    title="Intelligent Video Reframer",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include API routes
from app.api import routes
from app.api.web_routes import web_router
from fastapi.staticfiles import StaticFiles

app.include_router(routes.router, prefix="/api/v1")
app.include_router(web_router)

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")