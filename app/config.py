from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional
import os

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    # AWS S3
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: Optional[str] = None
    S3_PREFIX: str = ""

    # Google Gemini
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.5-pro"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Processing
    MAX_VIDEO_DURATION: int = 600  # 10 minutes
    MAX_VIDEO_SIZE: int = 500_000_000  # 500MB
    TEMP_DIR: str = "/tmp/video_processing"

    # Performance
    MAX_WORKERS: int = 4
    FRAME_ANALYSIS_FPS: float = 2.0  # Analyze 2 frames per second for music videos
    OUTPUT_VIDEO_QUALITY: str = "high"  # high, medium, fast

    # Railway specific
    PORT: int = 8080
    RAILWAY_ENVIRONMENT: Optional[str] = None

settings = Settings()