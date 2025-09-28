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

        # Calculate output dimensions (9:16 aspect ratio)
        input_width = metadata['width']
        input_height = metadata['height']
        # Calculate 9:16 aspect ratio, ensure even numbers for FFmpeg compatibility
        output_width = int(input_height * 9 / 16)
        output_height = input_height

        # Ensure even dimensions (FFmpeg requirement)
        if output_width % 2 != 0:
            output_width -= 1
        if output_height % 2 != 0:
            output_height -= 1

        logger.info(f"Output resolution: {output_width}x{output_height} (from {input_width}x{input_height})")

        # Ensure output width doesn't exceed input width
        if output_width > input_width:
            output_width = input_width
            output_height = int(input_width * 16 / 9)

        # Handle cuts vs smooth pans
        has_cuts = any(kf.is_cut for kf in keyframes)

        if has_cuts:
            # Complex processing with cuts
            output_path = await self.process_with_cuts(
                input_path, keyframes, output_width, output_height, quality
            )
        else:
            # Simple smooth crop with interpolation
            crop_filter = self.generate_smooth_crop_filter(
                keyframes, input_width, input_height, output_width, output_height
            )

            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-vf', crop_filter,
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
        out_w: int, out_h: int
    ) -> str:
        """Generate smooth crop filter using zoompan for interpolation"""

        if not keyframes:
            # Default center crop
            x = (in_w - out_w) // 2
            y = (in_h - out_h) // 2
            return f"crop={out_w}:{out_h}:{x}:{y}"

        # Use zoompan filter for smooth interpolation
        duration = keyframes[-1].timestamp if keyframes else 30

        # Build zoompan expression
        zoom_expr = "1"  # No zoom, just pan
        x_expr = self.build_position_expression([kf.center_x for kf in keyframes], [kf.timestamp for kf in keyframes], in_w, out_w)
        y_expr = self.build_position_expression([kf.center_y for kf in keyframes], [kf.timestamp for kf in keyframes], in_h, out_h)

        filter_str = (
            f"zoompan=z={zoom_expr}:x={x_expr}:y={y_expr}:"
            f"d=1:s={out_w}x{out_h}:fps=30"
        )

        return filter_str

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

                if segment['type'] == 'static':
                    # Static crop for this segment
                    await self.create_static_segment(
                        input_path, segment, segment_file, out_w, out_h, quality
                    )
                else:
                    # Smooth pan segment
                    await self.create_smooth_segment(
                        input_path, segment, segment_file, out_w, out_h, quality
                    )

                segment_files.append(segment_file)

            # Concatenate all segments
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
        out_w: int, out_h: int, quality: Dict
    ):
        """Create segment with static crop"""

        kf = segment['keyframes'][0]
        duration = segment['end_time'] - segment['start_time']

        # Calculate crop position
        metadata = await self.get_video_metadata(input_path)
        crop_x = int(kf.center_x * metadata['width'] - out_w / 2)
        crop_y = int(kf.center_y * metadata['height'] - out_h / 2)

        # Clamp to valid range
        crop_x = max(0, min(crop_x, metadata['width'] - out_w))
        crop_y = max(0, min(crop_y, metadata['height'] - out_h))

        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ss', str(segment['start_time']),
            '-t', str(duration),
            '-vf', f"crop={out_w}:{out_h}:{crop_x}:{crop_y}",
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
        out_w: int, out_h: int, quality: Dict
    ):
        """Create segment with smooth interpolation"""

        # Use the smooth crop filter for this segment
        metadata = await self.get_video_metadata(input_path)
        crop_filter = self.generate_smooth_crop_filter(
            segment['keyframes'],
            metadata['width'], metadata['height'],
            out_w, out_h
        )

        duration = segment['end_time'] - segment['start_time']

        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ss', str(segment['start_time']),
            '-t', str(duration),
            '-vf', crop_filter,
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

        return {
            'width': int(video_stream['width']),
            'height': int(video_stream['height']),
            'fps': eval(video_stream['r_frame_rate']),
            'duration': float(data['format']['duration']),
            'codec': video_stream['codec_name'],
            'bitrate': int(data['format'].get('bit_rate', 0))
        }

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