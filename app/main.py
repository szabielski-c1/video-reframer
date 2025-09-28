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