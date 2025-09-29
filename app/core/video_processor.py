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

            # Break down AI analysis into more visible steps
            await self.update_job_status(job_id, JobStatus.ANALYZING, 16.0, "Uploading video to Gemini AI")
            await asyncio.sleep(0.1)  # Small delay to show status update

            await self.update_job_status(job_id, JobStatus.ANALYZING, 20.0, "Video uploaded, creating analysis prompt")
            await asyncio.sleep(0.1)

            await self.update_job_status(job_id, JobStatus.ANALYZING, 22.0, "Gemini AI analyzing video content...")

            reframing_data = await self.gemini.analyze_video_for_reframing(local_path, request.settings, metadata)

            await self.update_job_status(job_id, JobStatus.ANALYZING, 35.0, "AI analysis complete, processing results")

            # Check if AI analysis was successful
            ai_confidence = reframing_data.get('confidence', 0)
            if ai_confidence < 0.05:
                raise Exception(f"AI reframing analysis failed with confidence {ai_confidence:.2f}. Gemini service may be unavailable or blocked by safety filters.")
            elif ai_confidence < 0.2:
                logger.warning(f"AI analysis has low confidence ({ai_confidence:.2f}) but proceeding anyway")

            # Align shots to frame boundaries for perfect timing
            await self.update_job_status(job_id, JobStatus.PROCESSING, 40.0, "Aligning shots to frame boundaries")
            fps = metadata.get('fps', 24.0)
            frame_count = metadata.get('frame_count', 0)

            # Validate fps to prevent division by zero
            if fps <= 0:
                logger.warning(f"Invalid fps value ({fps}), using default of 24.0")
                fps = 24.0

            await self.update_job_status(job_id, JobStatus.PROCESSING, 42.0, f"Processing {len(reframing_data['shots'])} shots at {fps:.1f} fps")
            reframing_data['shots'] = self.align_shots_to_frames(reframing_data['shots'], fps, frame_count, metadata['duration'])

            # Convert shot-based reframing data to crop keyframes
            await self.update_job_status(job_id, JobStatus.PROCESSING, 45.0, "Converting reframing plan to keyframes")
            crop_keyframes = self.convert_shots_to_keyframes(reframing_data['shots'], metadata)
            await self.update_job_status(job_id, JobStatus.PROCESSING, 50.0, f"Generated {len(crop_keyframes)} keyframes for smooth transitions")

            # Apply reframing with FFmpeg
            await self.update_job_status(job_id, JobStatus.PROCESSING, 55.0, "Starting video reframing with FFmpeg")

            has_cuts = any(kf.is_cut for kf in crop_keyframes)
            if has_cuts:
                await self.update_job_status(job_id, JobStatus.PROCESSING, 60.0, "Processing video with cuts and transitions")
            else:
                await self.update_job_status(job_id, JobStatus.PROCESSING, 60.0, "Applying smooth crop transitions")

            output_path = await self.ffmpeg.apply_reframing(local_path, crop_keyframes, request.settings, metadata)

            await self.update_job_status(job_id, JobStatus.PROCESSING, 85.0, "Video reframing complete")

            # Generate preview if requested
            preview_url = None
            if request.preview_only:
                await self.update_job_status(job_id, JobStatus.PROCESSING, 87.0, "Generating preview")
                preview_url = await self.generate_preview(local_path, crop_keyframes, job_id)

            # Upload to S3
            await self.update_job_status(job_id, JobStatus.UPLOADING, 90.0, "Uploading result to S3")
            output_url = await self.upload_result(output_path, request, job_id)
            logger.info(f"✅ Upload completed successfully. Output URL: {output_url}")
            await self.update_job_status(job_id, JobStatus.UPLOADING, 95.0, "Upload complete, finalizing")

            # Generate analytics
            await self.update_job_status(job_id, JobStatus.UPLOADING, 97.0, "Generating analytics")
            logger.info(f"🔍 Starting analytics generation for job {job_id}")
            analytics = self.generate_analytics_from_shots(reframing_data['shots'], crop_keyframes, metadata)
            logger.info(f"✅ Analytics generation completed for job {job_id}")

            # Complete job
            await self.update_job_status(job_id, JobStatus.UPLOADING, 99.0, "Finishing up")
            logger.info(f"🏁 Completing job {job_id} with output URL: {output_url}")
            try:
                await self.complete_job(job_id, output_url, preview_url, analytics)
                logger.info(f"✅ Job {job_id} completed successfully")
            except Exception as e:
                logger.error(f"❌ Failed to complete job {job_id}: {e}")
                raise

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

    def align_to_frame_boundary(self, timestamp: float, fps: float) -> float:
        """Align timestamp to nearest frame boundary"""
        if fps <= 0:
            logger.warning(f"Invalid fps ({fps}) in align_to_frame_boundary, returning original timestamp")
            return timestamp
        frame_number = round(timestamp * fps)
        return frame_number / fps

    def align_shots_to_frames(self, shots: List[Dict], fps: float, frame_count: int, total_duration: float) -> List[Dict]:
        """
        Align all shot boundaries to frame boundaries ensuring:
        1. Frame-perfect alignment (no flash frames)
        2. No missing frames (every frame is accounted for)
        3. Total duration matches exactly
        """
        if not shots:
            logger.warning("No shots to align")
            return shots

        logger.info(f"🎯 Aligning {len(shots)} shots to frame boundaries (fps={fps:.2f}, frames={frame_count}, duration={total_duration:.3f}s)")

        # Create aligned shots list
        aligned_shots = []

        for i, shot in enumerate(shots):
            aligned_shot = shot.copy()

            # Align shot boundaries to frame boundaries
            original_start = shot['start_time']
            original_end = shot['end_time']

            aligned_start = self.align_to_frame_boundary(original_start, fps)
            aligned_end = self.align_to_frame_boundary(original_end, fps)

            aligned_shot['start_time'] = aligned_start
            aligned_shot['end_time'] = aligned_end
            aligned_shot['duration'] = aligned_end - aligned_start

            # Align keyframe timestamps within the shot
            if 'keyframes' in shot:
                for keyframe in aligned_shot['keyframes']:
                    original_kf_time = keyframe.get('timestamp', aligned_start)
                    keyframe['timestamp'] = self.align_to_frame_boundary(original_kf_time, fps)

                    # Ensure keyframe timestamps are within shot boundaries
                    keyframe['timestamp'] = max(aligned_start, min(aligned_end, keyframe['timestamp']))

            logger.debug(f"Shot {i+1}: {original_start:.3f}s-{original_end:.3f}s → {aligned_start:.3f}s-{aligned_end:.3f}s (frames {int(aligned_start*fps)}-{int(aligned_end*fps)})")
            aligned_shots.append(aligned_shot)

        # Ensure complete coverage with no gaps or overlaps
        aligned_shots = self.ensure_complete_frame_coverage(aligned_shots, fps, frame_count, total_duration)

        # Validate frame coverage
        self.validate_frame_coverage(aligned_shots, fps, frame_count, total_duration)

        return aligned_shots

    def ensure_complete_frame_coverage(self, shots: List[Dict], fps: float, frame_count: int, total_duration: float) -> List[Dict]:
        """
        Ensure shots cover every frame with no gaps or overlaps
        """
        if not shots:
            return shots

        # Sort shots by start time
        shots.sort(key=lambda x: x['start_time'])

        # Ensure first shot starts at frame 0
        if shots[0]['start_time'] > 0:
            logger.info(f"📐 Extending first shot to start at frame 0 (was {shots[0]['start_time']:.3f}s)")
            shots[0]['start_time'] = 0.0
            shots[0]['duration'] = shots[0]['end_time'] - shots[0]['start_time']

        # Fix gaps and overlaps between shots
        for i in range(len(shots) - 1):
            current_shot = shots[i]
            next_shot = shots[i + 1]

            current_end_frame = round(current_shot['end_time'] * fps)
            next_start_frame = round(next_shot['start_time'] * fps)

            if current_end_frame < next_start_frame:
                # Gap detected - extend current shot to meet next shot
                gap_frames = next_start_frame - current_end_frame
                logger.info(f"📐 Closing {gap_frames} frame gap between shots {i+1} and {i+2}")
                current_shot['end_time'] = next_shot['start_time']
                current_shot['duration'] = current_shot['end_time'] - current_shot['start_time']

            elif current_end_frame > next_start_frame:
                # Overlap detected - trim current shot to meet next shot
                overlap_frames = current_end_frame - next_start_frame
                logger.info(f"📐 Removing {overlap_frames} frame overlap between shots {i+1} and {i+2}")
                current_shot['end_time'] = next_shot['start_time']
                current_shot['duration'] = current_shot['end_time'] - current_shot['start_time']

        # Ensure last shot ends exactly at the video duration
        last_shot = shots[-1]

        # Use frame count if valid, otherwise use total duration
        if frame_count > 0 and fps > 0:
            expected_end_time = frame_count / fps  # Frame-perfect end time
            logger.debug(f"Using frame-perfect end time: {frame_count} frames / {fps} fps = {expected_end_time:.3f}s")
        else:
            expected_end_time = total_duration  # Fallback to duration when frame count unavailable
            if frame_count <= 0:
                logger.warning(f"⚠️ Using total duration ({total_duration:.3f}s) as frame count is invalid ({frame_count})")
            else:
                logger.warning(f"⚠️ Using total duration ({total_duration:.3f}s) as fps is invalid ({fps})")

        if abs(last_shot['end_time'] - expected_end_time) > 1/fps:
            logger.info(f"📐 Adjusting last shot end time: {last_shot['end_time']:.3f}s → {expected_end_time:.3f}s")
            last_shot['end_time'] = expected_end_time
            last_shot['duration'] = last_shot['end_time'] - last_shot['start_time']

        return shots

    def validate_frame_coverage(self, shots: List[Dict], fps: float, frame_count: int, total_duration: float):
        """
        Validate that shots provide complete frame coverage
        """
        if not shots:
            raise Exception("No shots to validate")

        # Safety check for zero frame count or fps
        if frame_count <= 0:
            logger.warning(f"⚠️ Invalid frame count ({frame_count}), skipping frame validation")
            return

        if fps <= 0:
            logger.warning(f"⚠️ Invalid fps ({fps}), skipping frame validation")
            return

        # Check total coverage
        total_shot_duration = sum(shot['duration'] for shot in shots)
        expected_duration = frame_count / fps

        if abs(total_shot_duration - expected_duration) > 1/fps:
            raise Exception(f"Shot duration mismatch: {total_shot_duration:.3f}s vs expected {expected_duration:.3f}s")

        # Check for gaps
        shots.sort(key=lambda x: x['start_time'])
        for i in range(len(shots) - 1):
            current_end = shots[i]['end_time']
            next_start = shots[i + 1]['start_time']

            if abs(current_end - next_start) > 1/(fps*2):  # Allow for sub-frame rounding
                raise Exception(f"Gap detected between shots {i+1} and {i+2}: {current_end:.3f}s to {next_start:.3f}s")

        # Check boundaries
        if shots[0]['start_time'] != 0.0:
            raise Exception(f"First shot doesn't start at 0.0s: {shots[0]['start_time']:.3f}s")

        expected_end = frame_count / fps
        if abs(shots[-1]['end_time'] - expected_end) > 1/(fps*2):
            raise Exception(f"Last shot doesn't end at expected time: {shots[-1]['end_time']:.3f}s vs {expected_end:.3f}s")

        # Log validation success
        covered_frames = sum(round(shot['duration'] * fps) for shot in shots)
        logger.info(f"✅ Frame coverage validated: {len(shots)} shots covering {covered_frames}/{frame_count} frames ({covered_frames/frame_count*100:.1f}%)")

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

        # Create detailed shot breakdown for comparison
        gemini_shots = []
        for i, shot in enumerate(shots):
            crop_center = shot.get('crop_center', [0.5, 0.5])
            primary_subjects = shot.get('primary_subjects', [])

            gemini_shots.append({
                'shot_number': i + 1,
                'start_time': round(shot['start_time'], 3),
                'end_time': round(shot['end_time'], 3),
                'duration': round(shot['duration'], 3),
                'strategy': shot.get('crop_strategy', 'unknown'),
                'confidence': round(shot.get('confidence', 0), 2),
                'description': shot.get('shot_description', '')[:80],  # Show more description
                'crop_center_x': round(crop_center[0], 3) if len(crop_center) > 0 else 0.5,
                'crop_center_y': round(crop_center[1], 3) if len(crop_center) > 1 else 0.5,
                'primary_subjects': primary_subjects[:3] if primary_subjects else [],  # Show up to 3 subjects
                'reasoning': self.generate_crop_reasoning(shot, crop_center, primary_subjects)
            })

        # Create keyframe breakdown to show our processing
        our_keyframes = []
        for i, kf in enumerate(keyframes):
            our_keyframes.append({
                'keyframe_number': i + 1,
                'timestamp': round(kf.timestamp, 3),
                'center_x': round(kf.center_x, 3),
                'center_y': round(kf.center_y, 3),
                'is_cut': kf.is_cut,
                'confidence': round(kf.confidence, 2),
                'reason': kf.reason[:50]  # Truncate for display
            })

        return {
            'input_duration': metadata['duration'],
            'input_resolution': f"{metadata['width']}x{metadata['height']}",
            'output_resolution': f"{int(metadata['height'] * 9/16)}x{metadata['height']}",
            'frames_analyzed': len(shots),  # Frontend expects 'frames_analyzed'
            'shots_analyzed': len(shots),   # Keep this for backwards compatibility
            'total_keyframes': len(keyframes),
            'total_cuts': total_cuts,
            'average_confidence': avg_confidence,
            'subject_statistics': shot_stats,  # Frontend expects 'subject_statistics'
            'shot_statistics': shot_stats,     # Keep this for backwards compatibility
            'processing_approach': 'video_upload_analysis',
            'processing_time': (datetime.utcnow() - datetime.utcnow()).total_seconds(),  # Would track actual time
            # New detailed comparison data
            'gemini_shots': gemini_shots,
            'our_keyframes': our_keyframes,
            'shot_detection_comparison': {
                'gemini_shot_count': len(shots),
                'our_keyframe_count': len(keyframes),
                'cut_count': total_cuts,
                'avg_shot_duration': round(sum(shot['duration'] for shot in shots) / len(shots) if shots else 0, 3),
                'shortest_shot': round(min(shot['duration'] for shot in shots) if shots else 0, 3),
                'longest_shot': round(max(shot['duration'] for shot in shots) if shots else 0, 3)
            }
        }

    def generate_crop_reasoning(self, shot: Dict, crop_center: List[float], primary_subjects: List[str]) -> str:
        """Generate human-readable reasoning for crop center choice"""

        center_x, center_y = crop_center[0] if len(crop_center) > 0 else 0.5, crop_center[1] if len(crop_center) > 1 else 0.5
        strategy = shot.get('crop_strategy', 'unknown')

        # Determine position description
        if center_x < 0.3:
            h_pos = "left"
        elif center_x > 0.7:
            h_pos = "right"
        else:
            h_pos = "center"

        if center_y < 0.3:
            v_pos = "top"
        elif center_y > 0.7:
            v_pos = "bottom"
        else:
            v_pos = "middle"

        position = f"{v_pos}-{h_pos}" if v_pos != "middle" or h_pos != "center" else "center"

        # Build reasoning based on strategy and subjects
        reasoning_parts = []

        if primary_subjects:
            subject_text = ", ".join(primary_subjects[:2])
            reasoning_parts.append(f"Focus on {subject_text}")

        if strategy == "follow_subject":
            reasoning_parts.append("tracking movement")
        elif strategy == "track_speaker":
            reasoning_parts.append("keeping speaker in frame")
        elif strategy == "static_center":
            reasoning_parts.append("static composition")
        elif strategy == "pan_left_to_right":
            reasoning_parts.append("following camera pan")
        elif strategy == "zoom_in" or strategy == "zoom_out":
            reasoning_parts.append("handling zoom movement")

        reasoning_parts.append(f"positioned {position}")

        return "; ".join(reasoning_parts)

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
            # Fallback: put directly in reframe folder with job ID
            output_key = f"reframe/{job_id}_reframed.mp4"

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

