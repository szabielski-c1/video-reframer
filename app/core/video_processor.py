import asyncio
import logging
import json
import tempfile
import os
from typing import List, Dict, Optional
import uuid
from datetime import datetime
import numpy as np

from app.models import *
from app.config import settings
from app.services.s3_service import S3Service
from app.services.ffmpeg_service import FFmpegService
from app.services.gemini_service import GeminiService
from app.core.frame_analyzer import FrameAnalyzer
from app.core.trajectory_planner import TrajectoryPlanner
from app.core.subject_tracker import SubjectTracker
from app.utils.video_utils import extract_frames, get_video_metadata

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.s3 = S3Service()
        self.ffmpeg = FFmpegService()
        self.gemini = GeminiService()
        self.analyzer = FrameAnalyzer(self.gemini)
        self.trajectory_planner = TrajectoryPlanner()
        self.subject_tracker = SubjectTracker()
        self.current_job_id = None

    async def process_video(self, job_id: str, request: ReframeRequest):
        """Main processing pipeline"""
        self.current_job_id = job_id
        local_path = None
        output_path = None

        try:
            await self.update_job_status(job_id, JobStatus.DOWNLOADING, 0.0, "Downloading video")

            # Download video to temp location
            local_path = await self.download_video(request.input_url, job_id)

            # Get video metadata
            metadata = await self.ffmpeg.get_video_metadata(local_path)
            self.validate_video(metadata)

            # Extract frames for analysis
            await self.update_job_status(job_id, JobStatus.ANALYZING, 10.0, "Extracting frames for analysis")
            frames = await self.extract_analysis_frames(local_path, metadata)

            # Analyze frames with Gemini
            await self.update_job_status(job_id, JobStatus.ANALYZING, 20.0, "Analyzing scenes with AI")
            frame_analyses = await self.analyze_frames(frames, request, metadata)

            # Check if AI analysis was successful
            ai_confidence = sum(a.confidence for a in frame_analyses) / len(frame_analyses) if frame_analyses else 0

            if ai_confidence < 0.2:  # Very low confidence indicates AI failure
                raise Exception(f"AI analysis failed with confidence {ai_confidence:.2f}. Gemini service may be unavailable or blocked by safety filters.")

            # Track subjects across frames
            frame_analyses = self.subject_tracker.track_subjects(frame_analyses)

            # Generate smooth crop trajectory
            await self.update_job_status(job_id, JobStatus.PROCESSING, 50.0, "Planning camera movements")
            crop_keyframes = self.trajectory_planner.plan_trajectory(frame_analyses, request.settings, metadata)

            # Apply reframing with FFmpeg
            await self.update_job_status(job_id, JobStatus.PROCESSING, 70.0, "Reframing video")
            output_path = await self.ffmpeg.apply_reframing(local_path, crop_keyframes, request.settings, metadata)

            # Generate preview if requested
            preview_url = None
            if request.preview_only:
                preview_url = await self.generate_preview(local_path, crop_keyframes, job_id)

            # Upload to S3
            await self.update_job_status(job_id, JobStatus.UPLOADING, 90.0, "Uploading result")
            output_url = await self.upload_result(output_path, request, job_id)

            # Generate analytics
            analytics = self.generate_analytics(frame_analyses, crop_keyframes, metadata)

            # Complete job
            await self.complete_job(job_id, output_url, preview_url, analytics)

            # Send webhook if provided
            if request.webhook_url:
                await self.send_webhook(request.webhook_url, job_id, output_url)

        except Exception as e:
            error_message = self.categorize_error(e)
            logger.error(f"Error processing job {job_id}: {error_message}")
            await self.fail_job(job_id, error_message)
            raise

        finally:
            # Cleanup temp files
            self.cleanup_temp_files([local_path, output_path])

    async def update_job_status(self, job_id: str, status: JobStatus, progress: float, message: str):
        """Update job status in Redis"""
        await self.redis.hset(f"job:{job_id}", mapping={
            'status': status.value,
            'progress': progress,
            'message': message,
            'updated_at': datetime.utcnow().isoformat()
        })

    async def download_video(self, input_url: str, job_id: str) -> str:
        """Download video from S3 to temp location"""
        temp_path = os.path.join(tempfile.mkdtemp(), f"{job_id}_input.mp4")
        await self.s3.download_file(input_url, temp_path)
        return temp_path

    def validate_video(self, metadata: Dict):
        """Validate video meets processing requirements"""
        if metadata['duration'] > settings.MAX_VIDEO_DURATION:
            raise ValueError(f"Video duration {metadata['duration']}s exceeds maximum {settings.MAX_VIDEO_DURATION}s")

        if metadata['width'] / metadata['height'] != 16/9:
            logger.warning(f"Video aspect ratio is not 16:9, got {metadata['width']}x{metadata['height']}")

    async def extract_analysis_frames(self, video_path: str, metadata: Dict) -> List[np.ndarray]:
        """Extract frames for Gemini analysis at consistent 2 FPS for music videos"""
        frame_paths = await self.ffmpeg.extract_frames(
            video_path,
            fps=settings.FRAME_ANALYSIS_FPS,
            duration=min(metadata['duration'], settings.MAX_VIDEO_DURATION)
        )

        frames = []
        for path in frame_paths:
            frame = await extract_frames(path)
            frames.append(frame)

        return frames

    async def analyze_frames(self, frames: List[np.ndarray], request: ReframeRequest, metadata: Dict) -> List[FrameAnalysis]:
        """Analyze frames using Gemini with intelligent batching"""

        # Determine scene type and complexity first
        scene_info = await self.analyzer.classify_scene(frames[:10], request)

        # Build comprehensive analysis
        analyses = []
        batch_size = 5  # Process 5 frames at once

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            timestamps = [i * (1.0 / settings.FRAME_ANALYSIS_FPS) for j in range(i, min(i+batch_size, len(frames)))]

            # Analyze batch with context
            batch_analyses = await self.analyzer.analyze_batch(
                batch,
                timestamps,
                scene_info,
                request,
                previous_analyses=analyses[-5:] if analyses else None
            )

            analyses.extend(batch_analyses)

            # Update progress
            progress = 20 + (30 * len(analyses) / len(frames))
            await self.update_job_status(
                self.current_job_id,
                JobStatus.ANALYZING,
                progress,
                f"Analyzed {len(analyses)}/{len(frames)} frames"
            )

        return analyses

    async def generate_preview(self, video_path: str, keyframes: List[CropKeyframe], job_id: str) -> str:
        """Generate preview video with overlay"""
        preview_path = tempfile.mktemp(suffix='.mp4')
        await self.ffmpeg.generate_preview_overlay(video_path, keyframes, preview_path)

        # Upload preview to S3
        preview_key = f"previews/{job_id}_preview.mp4"
        preview_url = await self.s3.upload_file(preview_path, preview_key)

        return preview_url

    async def upload_result(self, output_path: str, request: ReframeRequest, job_id: str) -> str:
        """Upload processed video to S3"""
        if request.output_key:
            output_key = request.output_key
        else:
            output_key = f"output/{job_id}_reframed.mp4"

        output_url = await self.s3.upload_file(output_path, output_key, bucket=request.output_bucket)
        return output_url

    def generate_analytics(self, analyses: List[FrameAnalysis], keyframes: List[CropKeyframe], metadata: Dict) -> Dict:
        """Generate processing analytics"""
        total_cuts = sum(1 for kf in keyframes if kf.is_cut)
        avg_confidence = sum(a.confidence for a in analyses) / len(analyses) if analyses else 0

        subject_stats = {}
        for analysis in analyses:
            for subject in analysis.subjects:
                if subject.id not in subject_stats:
                    subject_stats[subject.id] = {
                        'screen_time': 0,
                        'speaking_time': 0,
                        'primary_count': 0
                    }
                subject_stats[subject.id]['screen_time'] += 1
                if subject.is_speaking:
                    subject_stats[subject.id]['speaking_time'] += 1
                if analysis.primary_subject == subject.id:
                    subject_stats[subject.id]['primary_count'] += 1

        return {
            'input_duration': metadata['duration'],
            'input_resolution': f"{metadata['width']}x{metadata['height']}",
            'output_resolution': f"{int(metadata['height'] * 9/16)}x{metadata['height']}",
            'frames_analyzed': len(analyses),
            'total_keyframes': len(keyframes),
            'total_cuts': total_cuts,
            'average_confidence': avg_confidence,
            'subject_statistics': subject_stats,
            'processing_time': (datetime.utcnow() - datetime.utcnow()).total_seconds()  # Would track actual time
        }

    async def complete_job(self, job_id: str, output_url: str, preview_url: Optional[str], analytics: Dict):
        """Mark job as completed"""
        await self.redis.hset(f"job:{job_id}", mapping={
            'status': JobStatus.COMPLETED.value,
            'progress': 100.0,
            'message': 'Processing completed successfully',
            'output_url': output_url,
            'preview_url': preview_url or '',
            'analytics': json.dumps(analytics),
            'updated_at': datetime.utcnow().isoformat()
        })

    async def fail_job(self, job_id: str, error: str):
        """Mark job as failed"""
        await self.redis.hset(f"job:{job_id}", mapping={
            'status': JobStatus.FAILED.value,
            'error': error,
            'updated_at': datetime.utcnow().isoformat()
        })

    async def send_webhook(self, webhook_url: str, job_id: str, output_url: str):
        """Send webhook notification"""
        import httpx
        async with httpx.AsyncClient() as client:
            try:
                await client.post(webhook_url, json={
                    'job_id': job_id,
                    'status': 'completed',
                    'output_url': output_url,
                    'timestamp': datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.error(f"Failed to send webhook: {e}")

    def cleanup_temp_files(self, paths: List[Optional[str]]):
        """Clean up temporary files"""
        for path in paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    # Also remove parent temp directory if empty
                    parent = os.path.dirname(path)
                    if parent.startswith('/tmp/') and not os.listdir(parent):
                        os.rmdir(parent)
                except Exception as e:
                    logger.warning(f"Failed to clean up {path}: {e}")

    def categorize_error(self, error: Exception) -> str:
        """Categorize errors into user-friendly messages"""

        error_str = str(error).lower()

        # Gemini AI specific errors
        if "finish_reason" in error_str and "2" in error_str:
            return "AI safety filters blocked this video content. Try a different video or contact support."
        elif "quota" in error_str or "limit" in error_str:
            return "AI service quota exceeded. Please try again later or contact support."
        elif "not found" in error_str and "model" in error_str:
            return "AI model configuration error. Please contact support."
        elif "json" in error_str and ("parsing" in error_str or "decode" in error_str):
            return "AI response format error. This may be due to content filtering. Try a different video."

        # Video/FFmpeg errors
        elif "invalid" in error_str and ("format" in error_str or "codec" in error_str):
            return "Unsupported video format. Please use MP4, AVI, or MOV files."
        elif "permission" in error_str or "access" in error_str:
            return "File access error. Please try uploading the video again."
        elif "disk" in error_str or "space" in error_str:
            return "Insufficient storage space. Please try again later."

        # Network/S3 errors
        elif "connection" in error_str or "network" in error_str:
            return "Network connection error. Please check your internet and try again."
        elif "timeout" in error_str:
            return "Request timeout. Your video may be too large. Try a shorter video."
        elif "s3" in error_str or "bucket" in error_str:
            return "Storage service error. Please try again later or contact support."

        # Validation errors
        elif "duration" in error_str and "exceeds" in error_str:
            return f"Video too long. Maximum duration is {settings.MAX_VIDEO_DURATION} seconds."
        elif "size" in error_str and "exceeds" in error_str:
            return f"Video file too large. Maximum size is {settings.MAX_VIDEO_SIZE / 1024 / 1024:.0f}MB."
        elif "aspect ratio" in error_str:
            return "Video must have 16:9 aspect ratio for optimal results."

        # Generic fallback
        else:
            return f"Processing error: {str(error)[:100]}... Please try again or contact support."

