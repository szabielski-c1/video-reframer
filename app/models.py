from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Literal, Tuple
from datetime import datetime
from enum import Enum

class ProcessingMode(str, Enum):
    AUTO = "auto"
    FACE_PRIORITY = "face_priority"
    ACTION = "action"
    SPEAKER = "speaker"
    CUSTOM = "custom"

class VideoStyle(str, Enum):
    DOCUMENTARY = "documentary"
    VLOG = "vlog"
    SPORTS = "sports"
    PRESENTATION = "presentation"
    INTERVIEW = "interview"
    MUSIC = "music"
    AUTO = "auto"

class QualityPreset(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    FAST = "fast"

class ReframeSettings(BaseModel):
    mode: ProcessingMode = ProcessingMode.AUTO
    style: VideoStyle = VideoStyle.AUTO
    smoothing: float = Field(default=0.8, ge=0.0, le=1.0)
    padding: float = Field(default=1.2, ge=1.0, le=2.0)
    quality: QualityPreset = QualityPreset.HIGH
    min_hold_time: float = Field(default=2.0, ge=0.5, le=5.0)
    edge_padding: float = Field(default=0.1, ge=0.0, le=0.3)
    cut_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_cuts: bool = True
    preserve_text: bool = True
    audio_analysis: bool = True

class GeminiPrompts(BaseModel):
    custom_focus: Optional[str] = None
    exclude_areas: Optional[List[str]] = None
    priority_subjects: Optional[List[str]] = None

class ReframeRequest(BaseModel):
    input_url: str
    output_bucket: Optional[str] = None
    output_key: Optional[str] = None
    settings: ReframeSettings = ReframeSettings()
    gemini_prompts: GeminiPrompts = GeminiPrompts()
    webhook_url: Optional[str] = None
    preview_only: bool = False

    @validator('input_url')
    def validate_s3_url(cls, v):
        if not v.startswith(('s3://', 'https://')):
            raise ValueError('Input URL must be an S3 URL')
        return v

class SubjectInfo(BaseModel):
    id: str
    type: Literal["person", "object", "text", "animal"]
    bbox: List[float]  # [x_center, y_center, width, height] normalized 0-1
    confidence: float = Field(ge=0.0, le=1.0)
    is_speaking: bool = False
    is_moving: bool = False
    importance_score: float = Field(ge=0.0, le=1.0)
    name_or_description: str

class FrameAnalysis(BaseModel):
    timestamp: float
    subjects: List[SubjectInfo]
    primary_subject: Optional[str]
    scene_type: VideoStyle
    motion_vectors: Dict[str, float]
    text_regions: List[Dict]
    suggested_center: Tuple[float, float]
    confidence: float
    action_description: str
    audio_peak: Optional[float]
    should_cut_here: bool = False
    hold_duration_suggestion: float = 2.0

class CropKeyframe(BaseModel):
    timestamp: float
    center_x: float
    center_y: float
    is_cut: bool = False
    confidence: float
    reason: str

class JobStatus(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    ANALYZING = "analyzing"
    PROCESSING = "processing"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    progress: float = 0.0
    message: Optional[str] = None
    output_url: Optional[str] = None
    preview_url: Optional[str] = None
    analytics: Optional[Dict] = None
    error: Optional[str] = None