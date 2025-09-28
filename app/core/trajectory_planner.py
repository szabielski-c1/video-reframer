import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy import interpolate
from scipy.signal import savgol_filter
import logging

from app.models import FrameAnalysis, CropKeyframe, ReframeSettings, ProcessingMode

logger = logging.getLogger(__name__)

class TrajectoryPlanner:
    """Plans smooth camera movements between subjects for 9:16 reframing"""

    def __init__(self):
        self.aspect_ratio = 9/16  # Target aspect ratio
        self.frame_width = 1080    # Output width
        self.frame_height = 1920   # Output height

    def plan_trajectory(
        self,
        analyses: List[FrameAnalysis],
        settings: ReframeSettings,
        video_metadata: dict
    ) -> List[CropKeyframe]:
        """Generate smooth crop trajectory from frame analyses"""

        if not analyses:
            return self.generate_default_trajectory(video_metadata)

        # Extract key points from analyses
        key_points = self.extract_key_points(analyses, settings)

        # Identify segments (continuous pans vs cuts)
        segments = self.identify_segments(key_points, settings)

        # Generate keyframes for each segment
        crop_keyframes = []
        for segment in segments:
            if segment['type'] == 'cut':
                # Add a cut transition
                keyframe = CropKeyframe(
                    timestamp=segment['timestamp'],
                    center_x=segment['target_x'],
                    center_y=segment['target_y'],
                    is_cut=True,
                    confidence=segment['confidence'],
                    reason=segment['reason']
                )
                crop_keyframes.append(keyframe)
            else:
                # Generate smooth pan/drift
                smooth_path = self.generate_smooth_path(
                    segment['points'],
                    settings.smoothing,
                    video_metadata.get('fps', 30)
                )
                crop_keyframes.extend(smooth_path)

        # Apply edge padding constraints
        crop_keyframes = self.apply_edge_constraints(crop_keyframes, settings.edge_padding)

        # Optimize for minimum jerk
        if settings.smoothing > 0.5:
            crop_keyframes = self.minimize_jerk(crop_keyframes)

        return crop_keyframes

    def extract_key_points(self, analyses: List[FrameAnalysis], settings: ReframeSettings) -> List[dict]:
        """Extract important points for trajectory"""

        key_points = []
        last_primary = None
        last_switch_time = 0

        for i, analysis in enumerate(analyses):
            create_keypoint = False
            reason = ""

            # Check for subject switch
            if analysis.primary_subject != last_primary:
                time_since_switch = analysis.timestamp - last_switch_time

                if time_since_switch >= settings.min_hold_time:
                    create_keypoint = True
                    reason = f"Switch to {analysis.primary_subject}"
                    last_primary = analysis.primary_subject
                    last_switch_time = analysis.timestamp

            # Check for important events
            if analysis.confidence > 0.8 and (
                analysis.should_cut_here or
                (analysis.audio_peak and analysis.audio_peak > 0.8)
            ):
                create_keypoint = True
                reason = f"Important: {analysis.action_description[:30]}"

            # Regular sampling for smooth motion
            if i % int(settings.min_hold_time * 2) == 0:
                create_keypoint = True
                reason = "Regular checkpoint"

            # Add keypoint
            if create_keypoint:
                center = self.calculate_optimal_center(analysis, settings)

                key_points.append({
                    'timestamp': analysis.timestamp,
                    'x': center[0],
                    'y': center[1],
                    'primary_subject': analysis.primary_subject,
                    'confidence': analysis.confidence,
                    'reason': reason,
                    'should_cut': analysis.should_cut_here,
                    'hold_duration': analysis.hold_duration_suggestion
                })

        return key_points

    def calculate_optimal_center(self, analysis: FrameAnalysis, settings: ReframeSettings) -> Tuple[float, float]:
        """Calculate optimal crop center for 9:16 frame"""

        # Start with suggested center from Gemini
        center_x, center_y = analysis.suggested_center

        if analysis.subjects:
            # Find primary subject
            primary = None
            for subject in analysis.subjects:
                if subject.id == analysis.primary_subject:
                    primary = subject
                    break

            if not primary and analysis.subjects:
                # Fallback to highest importance score
                primary = max(analysis.subjects, key=lambda s: s.importance_score)

            if primary:
                # Use primary subject's center
                bbox = primary.bbox
                center_x = bbox[0]
                center_y = bbox[1]

                # Apply padding based on subject type
                padding = settings.padding

                # Adjust for movement
                if primary.is_moving and primary.id in analysis.motion_vectors:
                    # Lead the movement slightly
                    motion = analysis.motion_vectors[primary.id]
                    center_x += motion * padding * 0.2

                # Adjust for speaking (more headroom)
                if primary.type == 'person' and primary.is_speaking:
                    center_y -= 0.05  # Slight upward adjustment

                # Consider multiple subjects
                if len(analysis.subjects) > 1 and settings.mode == ProcessingMode.AUTO:
                    # Calculate weighted center
                    weighted_x = 0
                    weighted_y = 0
                    total_weight = 0

                    for subject in analysis.subjects[:3]:  # Consider top 3 subjects
                        weight = subject.importance_score
                        weighted_x += subject.bbox[0] * weight
                        weighted_y += subject.bbox[1] * weight
                        total_weight += weight

                    if total_weight > 0:
                        com_x = weighted_x / total_weight
                        com_y = weighted_y / total_weight

                        # Blend primary with center of mass
                        blend = 0.2  # 80% primary, 20% group
                        center_x = center_x * (1 - blend) + com_x * blend
                        center_y = center_y * (1 - blend) + com_y * blend

        # Handle text preservation
        if settings.preserve_text and analysis.text_regions:
            center_x, center_y = self.adjust_for_text(
                center_x, center_y,
                analysis.text_regions,
                self.aspect_ratio
            )

        # Ensure within valid bounds for 9:16 crop
        # X needs to fit the vertical crop within horizontal frame
        min_x = self.aspect_ratio / 2
        max_x = 1 - (self.aspect_ratio / 2)
        center_x = np.clip(center_x, min_x, max_x)

        # Y is typically centered for 9:16 but allow slight adjustment
        center_y = np.clip(center_y, 0.45, 0.55)

        return (center_x, center_y)

    def adjust_for_text(self, cx: float, cy: float, text_regions: List[Dict], aspect_ratio: float) -> Tuple[float, float]:
        """Adjust center to preserve important text"""

        for text in text_regions:
            if text.get('importance') == 'high':
                bbox = text['bbox']
                text_cx = bbox[0] + bbox[2] / 2
                text_cy = bbox[1] + bbox[3] / 2

                # Check if text would be cut off
                crop_left = cx - (aspect_ratio / 2)
                crop_right = cx + (aspect_ratio / 2)

                if bbox[0] < crop_left:
                    # Text on left edge, shift right
                    cx += min(0.1, crop_left - bbox[0])
                elif bbox[0] + bbox[2] > crop_right:
                    # Text on right edge, shift left
                    cx -= min(0.1, (bbox[0] + bbox[2]) - crop_right)

        return cx, cy

    def identify_segments(self, key_points: List[dict], settings: ReframeSettings) -> List[dict]:
        """Identify segments that should be cuts vs pans"""

        if not key_points:
            return []

        segments = []
        current_segment = {'type': 'pan', 'points': [key_points[0]], 'confidence': 1.0}

        for i in range(1, len(key_points)):
            point = key_points[i]
            prev_point = key_points[i-1]

            # Calculate distance and time delta
            distance = np.sqrt(
                (point['x'] - prev_point['x'])**2 +
                (point['y'] - prev_point['y'])**2
            )
            time_delta = point['timestamp'] - prev_point['timestamp']

            # Determine if we should cut
            should_cut = False
            reason = ""

            if settings.enable_cuts:
                # Explicit cut request
                if point.get('should_cut'):
                    should_cut = True
                    reason = "Scene change"

                # Distance threshold exceeded
                elif distance > settings.cut_threshold:
                    should_cut = True
                    reason = f"Distance: {distance:.2f}"

                # Pan would be too fast
                elif time_delta > 0:
                    pan_speed = distance / time_delta
                    if pan_speed > 0.5:
                        should_cut = True
                        reason = f"Speed: {pan_speed:.2f}"

                # Different subjects far apart
                elif (point['primary_subject'] != prev_point['primary_subject'] and
                      distance > 0.3):
                    should_cut = True
                    reason = "Subject switch"

            if should_cut:
                # Save current segment
                if current_segment['points']:
                    segments.append(current_segment)

                # Add cut
                segments.append({
                    'type': 'cut',
                    'timestamp': point['timestamp'],
                    'target_x': point['x'],
                    'target_y': point['y'],
                    'confidence': point['confidence'],
                    'reason': reason
                })

                # Start new segment
                current_segment = {
                    'type': 'pan',
                    'points': [point],
                    'confidence': point['confidence']
                }
            else:
                current_segment['points'].append(point)

        # Add final segment
        if current_segment['points']:
            segments.append(current_segment)

        return segments

    def generate_smooth_path(self, points: List[dict], smoothing: float, fps: float) -> List[CropKeyframe]:
        """Generate smooth interpolated path between points"""

        if len(points) < 2:
            return [CropKeyframe(
                timestamp=points[0]['timestamp'],
                center_x=points[0]['x'],
                center_y=points[0]['y'],
                confidence=points[0]['confidence'],
                reason=points[0]['reason']
            )]

        # Extract coordinates
        timestamps = [p['timestamp'] for p in points]
        x_coords = [p['x'] for p in points]
        y_coords = [p['y'] for p in points]

        # Generate interpolation timestamps
        duration = timestamps[-1] - timestamps[0]
        num_frames = max(2, int(duration * fps))
        interp_timestamps = np.linspace(timestamps[0], timestamps[-1], num_frames)

        # Interpolate positions
        if len(points) > 3 and smoothing > 0:
            # Cubic spline for smooth curves
            try:
                x_spline = interpolate.CubicSpline(timestamps, x_coords, bc_type='natural')
                y_spline = interpolate.CubicSpline(timestamps, y_coords, bc_type='natural')

                interp_x = x_spline(interp_timestamps)
                interp_y = y_spline(interp_timestamps)

                # Apply smoothing filter
                if len(interp_x) > 5:
                    window_size = min(int(fps * smoothing), len(interp_x) - 1)
                    if window_size % 2 == 0:
                        window_size += 1
                    window_size = max(5, window_size)

                    interp_x = savgol_filter(interp_x, window_size, 3)
                    interp_y = savgol_filter(interp_y, window_size, 3)
            except Exception as e:
                logger.warning(f"Spline interpolation failed, using linear: {e}")
                interp_x = np.interp(interp_timestamps, timestamps, x_coords)
                interp_y = np.interp(interp_timestamps, timestamps, y_coords)
        else:
            # Linear interpolation for few points
            interp_x = np.interp(interp_timestamps, timestamps, x_coords)
            interp_y = np.interp(interp_timestamps, timestamps, y_coords)

        # Generate keyframes
        keyframes = []
        for i, t in enumerate(interp_timestamps):
            # Find closest original point for metadata
            closest_idx = np.argmin([abs(p['timestamp'] - t) for p in points])
            closest_point = points[closest_idx]

            keyframes.append(CropKeyframe(
                timestamp=t,
                center_x=float(interp_x[i]),
                center_y=float(interp_y[i]),
                confidence=closest_point['confidence'],
                reason=closest_point['reason']
            ))

        return keyframes

    def apply_edge_constraints(self, keyframes: List[CropKeyframe], edge_padding: float) -> List[CropKeyframe]:
        """Apply edge padding constraints"""

        half_width = self.aspect_ratio / 2

        for kf in keyframes:
            # Apply horizontal constraints
            min_x = half_width + edge_padding
            max_x = 1 - half_width - edge_padding
            kf.center_x = np.clip(kf.center_x, min_x, max_x)

            # Vertical constraints (slight adjustment allowed)
            kf.center_y = np.clip(kf.center_y, 0.45, 0.55)

        return keyframes

    def minimize_jerk(self, keyframes: List[CropKeyframe]) -> List[CropKeyframe]:
        """Apply jerk minimization for ultra-smooth motion"""

        if len(keyframes) < 5:
            return keyframes

        # Extract positions
        x_positions = [kf.center_x for kf in keyframes]
        y_positions = [kf.center_y for kf in keyframes]

        # Apply minimum jerk filter (5-point weighted average)
        smoothed_x = self.apply_minimum_jerk_filter(x_positions)
        smoothed_y = self.apply_minimum_jerk_filter(y_positions)

        # Update keyframes (except cuts)
        for i, kf in enumerate(keyframes):
            if not kf.is_cut:
                kf.center_x = smoothed_x[i]
                kf.center_y = smoothed_y[i]

        return keyframes

    def apply_minimum_jerk_filter(self, positions: List[float]) -> List[float]:
        """Apply 5-point minimum jerk trajectory filter"""

        if len(positions) < 5:
            return positions

        smoothed = []
        weights = [0.06, 0.24, 0.4, 0.24, 0.06]  # Gaussian-like weights

        for i in range(len(positions)):
            if i < 2 or i >= len(positions) - 2:
                smoothed.append(positions[i])
            else:
                # Apply weighted average
                window = positions[i-2:i+3]
                if len(window) == 5:
                    value = sum(w * p for w, p in zip(weights, window))
                    smoothed.append(value)
                else:
                    smoothed.append(positions[i])

        return smoothed

    def generate_default_trajectory(self, metadata: dict) -> List[CropKeyframe]:
        """Generate default center crop when no analyses available"""

        duration = metadata.get('duration', 10)
        fps = metadata.get('fps', 30)
        num_frames = int(duration * fps)

        keyframes = []
        for i in range(0, num_frames, int(fps)):  # One keyframe per second
            keyframes.append(CropKeyframe(
                timestamp=i / fps,
                center_x=0.5,
                center_y=0.5,
                is_cut=False,
                confidence=0.1,
                reason="Default center crop"
            ))

        return keyframes