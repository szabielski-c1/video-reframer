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
# Legacy components no longer needed with video-based analysis
# from app.core.frame_analyzer import FrameAnalyzer
# from app.core.trajectory_planner import TrajectoryPlanner
# from app.core.subject_tracker import SubjectTracker
from app.utils.video_utils import extract_frames, get_video_metadata

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.s3 = S3Service()
        self.ffmpeg = FFmpegService()
        self.gemini = GeminiService()
        # Legacy components removed - now using direct Gemini video analysis
        # self.analyzer = FrameAnalyzer(self.gemini)
        # self.trajectory_planner = TrajectoryPlanner()
        # self.subject_tracker = SubjectTracker()
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

            # Get comprehensive reframing plan from Gemini
            await self.update_job_status(job_id, JobStatus.ANALYZING, 15.0, "Getting reframing plan from AI")
            reframing_data = await self.gemini.analyze_video_for_reframing(local_path, request.settings, metadata)

            # Check if AI analysis was successful
            ai_confidence = reframing_data.get('confidence', 0)
            if ai_confidence < 0.2:
                raise Exception(f"AI reframing analysis failed with confidence {ai_confidence:.2f}. Gemini service may be unavailable or blocked by safety filters.")

            # Convert shot-based reframing data to crop keyframes
            await self.update_job_status(job_id, JobStatus.PROCESSING, 45.0, "Converting reframing plan to keyframes")
            crop_keyframes = self.convert_shots_to_keyframes(reframing_data['shots'], metadata)

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
            analytics = self.generate_analytics_from_shots(reframing_data['shots'], crop_keyframes, metadata)

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

    def convert_shots_to_keyframes(self, shots: List[Dict], metadata: Dict) -> List[CropKeyframe]:
        """Convert shot-based reframing data to crop keyframes for FFmpeg"""

        from app.models import CropKeyframe

        crop_keyframes = []

        for shot_idx, shot in enumerate(shots):
            keyframes = shot.get('keyframes', [])

            if not keyframes:
                # Create default keyframes for the shot
                crop_center = shot.get('crop_center', [0.5, 0.5])
                keyframes = [
                    {'timestamp': shot['start_time'], 'center': crop_center, 'zoom': 1.0},
                    {'timestamp': shot['end_time'], 'center': crop_center, 'zoom': 1.0}
                ]

            # Convert each keyframe to CropKeyframe
            for i, kf in enumerate(keyframes):
                timestamp = kf.get('timestamp', shot['start_time'] + i * 1.0)

                # Safely handle center - could be array or missing
                center = kf.get('center', [0.5, 0.5])
                if not isinstance(center, (list, tuple)) or len(center) < 2:
                    logger.warning(f"Invalid center format in keyframe: {center}, using default [0.5, 0.5]")
                    center = [0.5, 0.5]

                zoom = max(0.5, min(2.0, float(kf.get('zoom', 1.0))))  # Limit zoom to reasonable range

                # Ensure center values are within bounds
                center[0] = max(0.0, min(1.0, float(center[0])))
                center[1] = max(0.0, min(1.0, float(center[1])))

                # Calculate crop region for 9:16 output
                # Original is 16:9, target is 9:16 (56.25% of width)
                crop_width = 0.5625 / zoom  # 9/16 aspect ratio
                crop_height = 1.0 / zoom

                # Ensure crop doesn't go outside bounds
                left = max(0, min(1 - crop_width, center[0] - crop_width/2))
                top = max(0, min(1 - crop_height, center[1] - crop_height/2))

                crop_keyframe = CropKeyframe(
                    timestamp=timestamp,
                    center_x=center[0],
                    center_y=center[1],
                    is_cut=(i == 0 and shot.get('transition_to_next') == 'cut'),
                    confidence=shot.get('confidence', 0.8),
                    reason=kf.get('description', f"Shot {shot_idx + 1} keyframe {i + 1}")
                )

                crop_keyframes.append(crop_keyframe)

        # Sort by timestamp
        crop_keyframes.sort(key=lambda x: x.timestamp)

        logger.info(f"Generated {len(crop_keyframes)} crop keyframes from {len(shots)} shots")
        return crop_keyframes

    def generate_analytics_from_shots(self, shots: List[Dict], keyframes: List[CropKeyframe], metadata: Dict) -> Dict:
        """Generate processing analytics from shot-based analysis"""

        total_cuts = sum(1 for kf in keyframes if kf.is_cut)
        avg_confidence = sum(shot.get('confidence', 0) for shot in shots) / len(shots) if shots else 0

        shot_stats = {
            'total_shots': len(shots),
            'avg_shot_duration': sum(shot['duration'] for shot in shots) / len(shots) if shots else 0,
            'shot_strategies': {}
        }

        # Count different crop strategies
        for shot in shots:
            strategy = shot.get('crop_strategy', 'unknown')
            shot_stats['shot_strategies'][strategy] = shot_stats['shot_strategies'].get(strategy, 0) + 1

        return {
            'input_duration': metadata['duration'],
            'input_resolution': f"{metadata['width']}x{metadata['height']}",
            'output_resolution': f"{int(metadata['height'] * 9/16)}x{metadata['height']}",
            'shots_analyzed': len(shots),
            'total_keyframes': len(keyframes),
            'total_cuts': total_cuts,
            'average_confidence': avg_confidence,
            'shot_statistics': shot_stats,
            'processing_approach': 'video_upload_analysis',
            'processing_time': (datetime.utcnow() - datetime.utcnow()).total_seconds()  # Would track actual time
        }

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

    # Legacy method - kept for compatibility but replaced by generate_analytics_from_shots
    def generate_analytics(self, analyses: List[FrameAnalysis], keyframes: List[CropKeyframe], metadata: Dict) -> Dict:
        """Generate processing analytics (legacy method)"""
        return self.generate_analytics_from_shots([], keyframes, metadata)

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

