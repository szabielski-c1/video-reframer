import subprocess
import json
import os
import tempfile
import logging
from typing import List, Optional, Dict
import asyncio

from app.models import CropKeyframe, ReframeSettings, QualityPreset

logger = logging.getLogger(__name__)

class FFmpegService:
    """Handles all FFmpeg operations for video processing"""

    def __init__(self):
        self.quality_presets = {
            QualityPreset.HIGH: {
                'crf': 18,
                'preset': 'slow',
                'bitrate': '8M',
                'audio_bitrate': '192k'
            },
            QualityPreset.MEDIUM: {
                'crf': 23,
                'preset': 'medium',
                'bitrate': '4M',
                'audio_bitrate': '128k'
            },
            QualityPreset.FAST: {
                'crf': 28,
                'preset': 'veryfast',
                'bitrate': '2M',
                'audio_bitrate': '96k'
            }
        }

    async def apply_reframing(
        self,
        input_path: str,
        keyframes: List[CropKeyframe],
        settings: ReframeSettings,
        metadata: Dict
    ) -> str:
        """Apply dynamic reframing using FFmpeg with keyframe interpolation"""

        output_path = tempfile.mktemp(suffix='.mp4')
        quality = self.quality_presets[settings.quality]

        # Calculate output dimensions (9:16 aspect ratio for vertical video)
        input_width = metadata['width']
        input_height = metadata['height']

        # Target standard vertical resolution: 1080x1920 (9:16 aspect ratio)
        target_width = 1080
        target_height = 1920

        # First, calculate the crop dimensions to get 9:16 aspect ratio
        # For 1920x1080 input, we want to crop to 607x1080 (9:16), then scale to 1080x1920
        crop_width = int(input_height * 9 / 16)  # 1080 * 9/16 = 607
        crop_height = input_height  # Use full height: 1080

        # Ensure crop dimensions don't exceed input
        if crop_width > input_width:
            crop_width = input_width
            crop_height = int(input_width * 16 / 9)

        # Ensure even dimensions for cropping (FFmpeg requirement)
        if crop_width % 2 != 0:
            crop_width -= 1
        if crop_height % 2 != 0:
            crop_height -= 1

        # Final output will be scaled to standard vertical resolution
        output_width = target_width
        output_height = target_height

        logger.info(f"Crop to: {crop_width}x{crop_height}, then scale to: {output_width}x{output_height} (from {input_width}x{input_height})")

        # Handle cuts vs smooth pans
        has_cuts = any(kf.is_cut for kf in keyframes)

        if has_cuts:
            # Complex processing with cuts
            output_path = await self.process_with_cuts(
                input_path, keyframes, crop_width, crop_height, output_width, output_height, quality
            )
        else:
            # Simple smooth crop with interpolation
            crop_filter = self.generate_smooth_crop_filter(
                keyframes, input_width, input_height, crop_width, crop_height
            )

            # Add scaling after cropping to get final resolution
            video_filter = f"{crop_filter},scale={output_width}:{output_height}"

            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-vf', video_filter,
                '-c:v', 'libx264',
                '-crf', str(quality['crf']),
                '-preset', quality['preset'],
                '-maxrate', quality['bitrate'],
                '-bufsize', str(int(quality['bitrate'].rstrip('M')) * 2) + 'M',
                '-c:a', 'aac',
                '-b:a', quality['audio_bitrate'],
                '-movflags', '+faststart',
                '-y',
                output_path
            ]

            await self.execute_ffmpeg(cmd)

        return output_path

    def generate_smooth_crop_filter(
        self,
        keyframes: List[CropKeyframe],
        in_w: int, in_h: int,
        crop_w: int, crop_h: int
    ) -> str:
        """Generate smooth crop filter to crop input to 9:16 aspect ratio"""

        if not keyframes:
            # Default center crop
            x = (in_w - crop_w) // 2
            y = (in_h - crop_h) // 2
            return f"crop={crop_w}:{crop_h}:{x}:{y}"

        # For now, use the first keyframe's position for a static crop
        # TODO: Implement smooth interpolation later once basic processing works
        first_kf = keyframes[0]

        # Convert normalized position to pixel position
        crop_x = int(first_kf.center_x * in_w - crop_w / 2)
        crop_y = int(first_kf.center_y * in_h - crop_h / 2)

        # Clamp to valid range
        crop_x = max(0, min(crop_x, in_w - crop_w))
        crop_y = max(0, min(crop_y, in_h - crop_h))

        logger.info(f"Using static crop at ({crop_x}, {crop_y}) for {crop_w}x{crop_h} crop")
        return f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}"

    def build_position_expression(self, positions: List[float], timestamps: List[float], input_size: int, output_size: int) -> str:
        """Build FFmpeg expression for position interpolation"""

        if len(positions) == 1:
            # Static position
            pixel_pos = max(0, min(positions[0] * input_size - output_size / 2, input_size - output_size))
            return str(int(pixel_pos))

        # Build piecewise linear interpolation expression
        expr_parts = []

        for i in range(len(positions)):
            pos = positions[i]
            time = timestamps[i]

            # Convert normalized position to pixel position
            pixel_pos = max(0, min(pos * input_size - output_size / 2, input_size - output_size))

            if i == 0:
                expr_parts.append(f"if(lt(t,{time}),{pixel_pos}")
            elif i == len(positions) - 1:
                prev_pos = max(0, min(positions[i-1] * input_size - output_size / 2, input_size - output_size))
                prev_time = timestamps[i-1]
                expr_parts.append(f",{prev_pos}+({pixel_pos}-{prev_pos})*(t-{prev_time})/({time}-{prev_time}))")
            else:
                prev_pos = max(0, min(positions[i-1] * input_size - output_size / 2, input_size - output_size))
                prev_time = timestamps[i-1]
                expr_parts.append(f",if(lt(t,{time}),{prev_pos}+({pixel_pos}-{prev_pos})*(t-{prev_time})/({time}-{prev_time})")

        # Close all if statements
        expr_parts.append(")" * (len(positions) - 1))

        # Fallback for times beyond keyframes
        final_pos = max(0, min(positions[-1] * input_size - output_size / 2, input_size - output_size))
        expr_parts.append(f",{final_pos})")

        return "".join(expr_parts)

    async def process_with_cuts(
        self,
        input_path: str,
        keyframes: List[CropKeyframe],
        crop_w: int, crop_h: int,
        out_w: int, out_h: int,
        quality: Dict
    ) -> str:
        """Process video with cuts by creating segments and concatenating"""

        # Split keyframes into segments
        segments = self.split_into_segments(keyframes)

        # Process each segment
        segment_files = []
        temp_dir = tempfile.mkdtemp()

        try:
            for i, segment in enumerate(segments):
                segment_file = os.path.join(temp_dir, f"segment_{i:03d}.mp4")

                try:
                    if segment['type'] == 'static':
                        # Static crop for this segment
                        await self.create_static_segment(
                            input_path, segment, segment_file, crop_w, crop_h, out_w, out_h, quality
                        )
                    else:
                        # Smooth pan segment
                        await self.create_smooth_segment(
                            input_path, segment, segment_file, crop_w, crop_h, out_w, out_h, quality
                        )

                    # Only add to list if file was actually created
                    if os.path.exists(segment_file) and os.path.getsize(segment_file) > 0:
                        segment_files.append(segment_file)
                        logger.info(f"✓ Created segment file: {os.path.basename(segment_file)} ({os.path.getsize(segment_file)} bytes)")
                    else:
                        logger.warning(f"⚠️ Segment file not created or empty: {segment_file}")

                except Exception as e:
                    logger.error(f"❌ Failed to create segment {i}: {e}")
                    continue

            # Concatenate all segments
            if not segment_files:
                raise Exception("No valid video segments were created")

            logger.info(f"🔗 Concatenating {len(segment_files)} valid segments")
            output_path = tempfile.mktemp(suffix='.mp4')
            await self.concatenate_segments(segment_files, output_path)

            return output_path

        finally:
            # Cleanup temp files
            for file in segment_files:
                if os.path.exists(file):
                    os.remove(file)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    def split_into_segments(self, keyframes: List[CropKeyframe]) -> List[Dict]:
        """Split keyframes into segments based on cuts"""

        segments = []
        current_segment = []

        for kf in keyframes:
            if kf.is_cut and current_segment:
                # End current segment
                segments.append({
                    'type': 'smooth',
                    'keyframes': current_segment,
                    'start_time': current_segment[0].timestamp,
                    'end_time': current_segment[-1].timestamp
                })
                current_segment = [kf]
            else:
                current_segment.append(kf)

        # Add final segment
        if current_segment:
            segments.append({
                'type': 'smooth',
                'keyframes': current_segment,
                'start_time': current_segment[0].timestamp,
                'end_time': current_segment[-1].timestamp
            })

        return segments

    async def create_static_segment(
        self, input_path: str, segment: Dict, output_path: str,
        crop_w: int, crop_h: int, out_w: int, out_h: int, quality: Dict
    ):
        """Create segment with static crop"""

        kf = segment['keyframes'][0]
        duration = segment['end_time'] - segment['start_time']

        # Ensure minimum duration to avoid zero-length segments
        if duration <= 0.01:
            logger.warning(f"Segment duration too short ({duration}s), skipping segment from {segment['start_time']} to {segment['end_time']}")
            return

        logger.info(f"Creating static segment: {segment['start_time']:.3f}s to {segment['end_time']:.3f}s (duration: {duration:.3f}s)")

        # Calculate crop position
        metadata = await self.get_video_metadata(input_path)
        crop_x = int(kf.center_x * metadata['width'] - crop_w / 2)
        crop_y = int(kf.center_y * metadata['height'] - crop_h / 2)

        # Clamp to valid range
        crop_x = max(0, min(crop_x, metadata['width'] - crop_w))
        crop_y = max(0, min(crop_y, metadata['height'] - crop_h))

        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ss', str(segment['start_time']),
            '-t', str(duration),
            '-vf', f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={out_w}:{out_h}",
            '-c:v', 'libx264',
            '-crf', str(quality['crf']),
            '-preset', quality['preset'],
            '-c:a', 'aac',
            '-b:a', quality['audio_bitrate'],
            '-y',
            output_path
        ]

        await self.execute_ffmpeg(cmd)

    async def create_smooth_segment(
        self, input_path: str, segment: Dict, output_path: str,
        crop_w: int, crop_h: int, out_w: int, out_h: int, quality: Dict
    ):
        """Create segment with smooth interpolation"""

        duration = segment['end_time'] - segment['start_time']

        # Ensure minimum duration to avoid zero-length segments
        if duration <= 0.01:
            logger.warning(f"Segment duration too short ({duration}s), skipping segment from {segment['start_time']} to {segment['end_time']}")
            return

        logger.info(f"Creating smooth segment: {segment['start_time']:.3f}s to {segment['end_time']:.3f}s (duration: {duration:.3f}s)")

        # Use the smooth crop filter for this segment
        metadata = await self.get_video_metadata(input_path)
        crop_filter = self.generate_smooth_crop_filter(
            segment['keyframes'],
            metadata['width'], metadata['height'],
            crop_w, crop_h
        )

        # Add scaling after cropping
        video_filter = f"{crop_filter},scale={out_w}:{out_h}"

        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ss', str(segment['start_time']),
            '-t', str(duration),
            '-vf', video_filter,
            '-c:v', 'libx264',
            '-crf', str(quality['crf']),
            '-preset', quality['preset'],
            '-c:a', 'aac',
            '-b:a', quality['audio_bitrate'],
            '-y',
            output_path
        ]

        await self.execute_ffmpeg(cmd)

    async def concatenate_segments(self, segment_files: List[str], output_path: str):
        """Concatenate video segments"""

        # Create concat file
        concat_file = tempfile.mktemp(suffix='.txt')
        with open(concat_file, 'w') as f:
            for file in segment_files:
                f.write(f"file '{file}'\n")

        try:
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',
                '-y',
                output_path
            ]

            await self.execute_ffmpeg(cmd)

        finally:
            if os.path.exists(concat_file):
                os.remove(concat_file)

    async def extract_frames(
        self,
        video_path: str,
        fps: float = 1.0,
        start_time: float = 0,
        duration: Optional[float] = None
    ) -> List[str]:
        """Extract frames from video at specified FPS"""

        output_dir = tempfile.mkdtemp()
        output_pattern = os.path.join(output_dir, 'frame_%04d.jpg')

        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-ss', str(start_time)
        ]

        if duration:
            cmd.extend(['-t', str(duration)])

        cmd.extend([
            '-vf', f'fps={fps}',
            '-q:v', '2',
            '-y',
            output_pattern
        ])

        await self.execute_ffmpeg(cmd)

        # Return sorted list of frame files
        frames = []
        for filename in sorted(os.listdir(output_dir)):
            if filename.endswith('.jpg'):
                frames.append(os.path.join(output_dir, filename))

        return frames

    async def get_video_metadata(self, video_path: str) -> Dict:
        """Extract video metadata using FFprobe"""

        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise Exception(f"FFprobe failed: {stderr.decode()}")

        data = json.loads(stdout)

        # Extract video stream info
        video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)

        if not video_stream:
            raise ValueError("No video stream found")

        # Parse frame rate more safely
        fps_string = video_stream.get('r_frame_rate', '24/1')
        if '/' in fps_string:
            numerator, denominator = fps_string.split('/')
            fps = float(numerator) / float(denominator) if float(denominator) != 0 else 24.0
        else:
            fps = float(fps_string) if fps_string else 24.0

        # Calculate frame count from duration and fps
        duration = float(data['format']['duration'])
        frame_count = int(duration * fps)

        # Try to get more accurate frame count from nb_frames if available
        if 'nb_frames' in video_stream:
            try:
                accurate_frame_count = int(video_stream['nb_frames'])
                if accurate_frame_count > 0:
                    logger.debug(f"Using nb_frames for accurate count: {accurate_frame_count} vs calculated {frame_count}")
                    frame_count = accurate_frame_count
            except (ValueError, TypeError):
                logger.debug("nb_frames not available or invalid, using calculated frame count")

        # Validate values
        if fps <= 0:
            logger.warning(f"Invalid fps ({fps}) detected, using default 24.0")
            fps = 24.0
            frame_count = int(duration * fps)

        if frame_count <= 0:
            logger.warning(f"Invalid frame count ({frame_count}) detected, recalculating")
            frame_count = int(duration * fps)

        # Final validation - if frame count is still suspect, try direct counting (slow but accurate)
        if frame_count <= 0 or (duration > 0 and abs(frame_count / fps - duration) > 1.0):
            logger.warning(f"Frame count seems incorrect ({frame_count}), attempting direct count...")
            try:
                direct_frame_count = await self.count_frames_directly(video_path)
                if direct_frame_count > 0:
                    logger.info(f"Direct frame count: {direct_frame_count} (was {frame_count})")
                    frame_count = direct_frame_count
            except Exception as e:
                logger.warning(f"Direct frame counting failed: {e}, keeping estimated count")

        logger.info(f"Video metadata: {int(video_stream['width'])}x{int(video_stream['height'])}, {fps:.2f}fps, {duration:.2f}s, {frame_count} frames")

        return {
            'width': int(video_stream['width']),
            'height': int(video_stream['height']),
            'fps': fps,
            'duration': duration,
            'frame_count': frame_count,
            'codec': video_stream['codec_name'],
            'bitrate': int(data['format'].get('bit_rate', 0))
        }

    async def count_frames_directly(self, video_path: str) -> int:
        """Count frames directly using ffprobe (slower but more accurate)"""

        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-count_frames',
            '-show_entries', 'stream=nb_frames',
            '-csv', '=',
            video_path
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise Exception(f"Frame counting failed: {stderr.decode()}")

        output = stdout.decode().strip()
        # Output format: "stream,1234" where 1234 is the frame count
        if ',' in output:
            return int(output.split(',')[1])

        return 0

    async def generate_preview_overlay(
        self,
        input_path: str,
        keyframes: List[CropKeyframe],
        output_path: str,
        preview_duration: float = 30.0
    ) -> str:
        """Generate preview video with crop area overlay"""

        metadata = await self.get_video_metadata(input_path)
        out_w = int(metadata['height'] * 9 / 16)
        out_h = metadata['height']

        # Build drawbox filter to show crop area
        drawbox_expressions = []
        for kf in keyframes:
            if kf.timestamp <= preview_duration:
                # Calculate crop position
                crop_x = int(kf.center_x * metadata['width'] - out_w / 2)
                crop_y = int(kf.center_y * metadata['height'] - out_h / 2)

                # Clamp to valid range
                crop_x = max(0, min(crop_x, metadata['width'] - out_w))
                crop_y = max(0, min(crop_y, metadata['height'] - out_h))

                # Create time-based drawbox
                drawbox_expressions.append(
                    f"drawbox=enable='between(t,{kf.timestamp},{kf.timestamp + 1})'"
                    f":x={crop_x}:y={crop_y}:w={out_w}:h={out_h}:color=red:thickness=3"
                )

        # Combine all drawbox filters
        filter_chain = ",".join(drawbox_expressions) if drawbox_expressions else "null"

        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-t', str(preview_duration),
            '-vf', f'{filter_chain},scale=1280:720',
            '-c:v', 'libx264',
            '-crf', '28',
            '-preset', 'veryfast',
            '-c:a', 'aac',
            '-b:a', '96k',
            '-y',
            output_path
        ]

        await self.execute_ffmpeg(cmd)

        return output_path

    async def execute_ffmpeg(self, cmd: List[str]):
        """Execute FFmpeg command with proper error handling"""

        logger.info(f"Executing FFmpeg: {' '.join(cmd[:10])}...")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown FFmpeg error"
            logger.error(f"FFmpeg error: {error_msg}")
            raise Exception(f"FFmpeg processing failed: {error_msg}")

        logger.info("FFmpeg processing completed successfully")
        return stdout.decode()