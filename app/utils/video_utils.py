import cv2
import numpy as np
import asyncio
import logging
import os
from typing import List, Dict, Tuple, Optional
from PIL import Image
import tempfile

logger = logging.getLogger(__name__)

async def extract_frames(image_path: str) -> np.ndarray:
    """Extract single frame as numpy array"""

    try:
        # Use PIL to load image
        img = Image.open(image_path)

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Convert to numpy array
        frame = np.array(img)

        return frame

    except Exception as e:
        logger.error(f"Failed to extract frame from {image_path}: {e}")
        raise

async def get_video_metadata(video_path: str) -> Dict:
    """Get video metadata using OpenCV"""

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate duration from frame count if available, otherwise use position
        if frame_count > 0 and fps > 0:
            duration = frame_count / fps
            logger.info(f"📊 Video metadata: {frame_count} frames at {fps:.2f}fps = {duration:.3f}s")
        else:
            logger.warning(f"⚠️ OpenCV frame count unavailable ({frame_count}), using fallback method")
            # Fallback: get duration by seeking to end
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
            duration = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            logger.info(f"📏 Fallback duration from seeking: {duration:.3f}s")

            # Calculate frame count from duration and fps
            if fps > 0 and duration > 0:
                frame_count = int(duration * fps)
                logger.info(f"🔢 Calculated frame count: {duration:.3f}s × {fps:.2f}fps = {frame_count} frames")
            else:
                logger.error(f"❌ Cannot calculate frame count: fps={fps}, duration={duration}")
                # Final fallback - use duration and default fps if fps is broken
                if duration > 0:
                    fallback_fps = fps if fps > 0 else 23.98  # Use detected fps or common default
                    frame_count = int(duration * fallback_fps)
                    logger.warning(f"🆘 Emergency fallback: {duration:.3f}s × {fallback_fps:.2f}fps = {frame_count} frames")
                    fps = fallback_fps  # Update fps for metadata

        cap.release()

        metadata = {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'aspect_ratio': width / height if height > 0 else 0,
            'file_size': os.path.getsize(video_path) if os.path.exists(video_path) else 0
        }

        return metadata

    except Exception as e:
        logger.error(f"Failed to get video metadata for {video_path}: {e}")
        raise

async def extract_video_frames(
    video_path: str,
    fps: float = 1.0,
    start_time: float = 0,
    duration: Optional[float] = None
) -> List[np.ndarray]:
    """Extract frames from video at specified FPS"""

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / video_fps if video_fps > 0 else 0

        # Calculate frame extraction parameters
        frame_interval = int(video_fps / fps) if fps < video_fps else 1
        start_frame = int(start_time * video_fps)

        if duration:
            end_frame = start_frame + int(duration * video_fps)
        else:
            end_frame = total_frames

        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        current_frame = start_frame

        while current_frame < end_frame:
            ret, frame = cap.read()

            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

            # Skip to next extraction point
            current_frame += frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        cap.release()

        logger.info(f"Extracted {len(frames)} frames from {video_path}")
        return frames

    except Exception as e:
        logger.error(f"Frame extraction failed for {video_path}: {e}")
        raise

def resize_frame(frame: np.ndarray, target_size: Tuple[int, int], maintain_aspect: bool = True) -> np.ndarray:
    """Resize frame to target size"""

    if maintain_aspect:
        # Calculate scaling to fit within target size
        h, w = frame.shape[:2]
        target_w, target_h = target_size

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize and pad if necessary
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Create padded image
        padded = np.zeros((target_h, target_w, frame.shape[2]), dtype=frame.dtype)

        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2

        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return padded
    else:
        # Simple resize without maintaining aspect ratio
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)

def crop_frame(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop frame to bounding box (x, y, width, height)"""

    x, y, w, h = bbox
    height, width = frame.shape[:2]

    # Clamp coordinates to frame boundaries
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = min(w, width - x)
    h = min(h, height - y)

    return frame[y:y+h, x:x+w]

def calculate_crop_params(
    frame_size: Tuple[int, int],
    center: Tuple[float, float],
    target_aspect: float
) -> Tuple[int, int, int, int]:
    """Calculate crop parameters for given center and aspect ratio"""

    width, height = frame_size
    center_x, center_y = center

    # Convert normalized coordinates to pixels
    center_x_px = int(center_x * width)
    center_y_px = int(center_y * height)

    # Calculate crop dimensions for target aspect ratio
    if target_aspect > (width / height):
        # Crop is wider than frame - use full height
        crop_height = height
        crop_width = int(crop_height * target_aspect)
    else:
        # Crop is taller than frame - use full width
        crop_width = width
        crop_height = int(crop_width / target_aspect)

    # Calculate crop position
    crop_x = center_x_px - crop_width // 2
    crop_y = center_y_px - crop_height // 2

    # Ensure crop stays within frame bounds
    crop_x = max(0, min(crop_x, width - crop_width))
    crop_y = max(0, min(crop_y, height - crop_height))

    return crop_x, crop_y, crop_width, crop_height

def detect_scene_changes(frames: List[np.ndarray], threshold: float = 0.3) -> List[int]:
    """Detect scene changes between frames using histogram comparison"""

    if len(frames) < 2:
        return []

    scene_changes = []

    for i in range(1, len(frames)):
        # Convert frames to grayscale for comparison
        gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)

        # Calculate histograms
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

        # Compare histograms using correlation
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        # If correlation is below threshold, it's a scene change
        if correlation < (1 - threshold):
            scene_changes.append(i)

    return scene_changes

def estimate_motion_vectors(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """Estimate motion vectors between two frames using optical flow"""

    try:
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)

        # Calculate average motion
        if flow is not None and len(flow) > 0:
            avg_motion_x = np.mean(flow[:, :, 0])
            avg_motion_y = np.mean(flow[:, :, 1])

            return np.array([avg_motion_x, avg_motion_y])
        else:
            return np.array([0.0, 0.0])

    except Exception as e:
        logger.warning(f"Motion estimation failed: {e}")
        return np.array([0.0, 0.0])

def apply_temporal_smoothing(values: List[float], window_size: int = 5) -> List[float]:
    """Apply temporal smoothing to a sequence of values"""

    if len(values) < window_size:
        return values.copy()

    smoothed = []
    half_window = window_size // 2

    for i in range(len(values)):
        # Calculate window bounds
        start = max(0, i - half_window)
        end = min(len(values), i + half_window + 1)

        # Calculate weighted average
        window_values = values[start:end]
        smoothed_value = sum(window_values) / len(window_values)
        smoothed.append(smoothed_value)

    return smoothed

def calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate difference between two frames"""

    try:
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)

        # Calculate mean difference
        mean_diff = np.mean(diff) / 255.0

        return mean_diff

    except Exception as e:
        logger.warning(f"Frame difference calculation failed: {e}")
        return 0.0

def extract_dominant_colors(frame: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
    """Extract dominant colors from frame using K-means clustering"""

    try:
        # Reshape frame for K-means
        data = frame.reshape((-1, 3))
        data = np.float32(data)

        # Apply K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert centers to integer RGB values
        centers = np.uint8(centers)
        dominant_colors = [tuple(color) for color in centers]

        return dominant_colors

    except Exception as e:
        logger.warning(f"Color extraction failed: {e}")
        return [(128, 128, 128)]  # Default gray

async def validate_video_file(file_path: str) -> bool:
    """Validate that file is a valid video"""

    try:
        if not os.path.exists(file_path):
            return False

        # Try to open with OpenCV
        cap = cv2.VideoCapture(file_path)
        is_valid = cap.isOpened()

        if is_valid:
            # Try to read first frame
            ret, _ = cap.read()
            is_valid = ret

        cap.release()
        return is_valid

    except Exception as e:
        logger.error(f"Video validation failed for {file_path}: {e}")
        return False

def create_video_thumbnail(frame: np.ndarray, size: Tuple[int, int] = (320, 180)) -> np.ndarray:
    """Create thumbnail from video frame"""

    try:
        # Resize maintaining aspect ratio
        thumbnail = resize_frame(frame, size, maintain_aspect=True)

        return thumbnail

    except Exception as e:
        logger.error(f"Thumbnail creation failed: {e}")
        return frame

def analyze_frame_composition(frame: np.ndarray) -> Dict:
    """Analyze frame composition using rule of thirds"""

    height, width = frame.shape[:2]

    # Rule of thirds grid points
    third_w = width // 3
    third_h = height // 3

    grid_points = [
        (third_w, third_h),      # Top-left
        (2 * third_w, third_h),  # Top-right
        (third_w, 2 * third_h),  # Bottom-left
        (2 * third_w, 2 * third_h)  # Bottom-right
    ]

    # Calculate interest points based on gradients
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Sobel gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Find areas of high interest (high gradients)
    interest_map = gradient_magnitude > np.percentile(gradient_magnitude, 80)

    # Calculate composition score for each grid point
    composition_scores = []
    for point in grid_points:
        x, y = point
        # Sample area around grid point
        sample_size = 50
        x1 = max(0, x - sample_size)
        x2 = min(width, x + sample_size)
        y1 = max(0, y - sample_size)
        y2 = min(height, y + sample_size)

        sample_area = interest_map[y1:y2, x1:x2]
        score = np.mean(sample_area)
        composition_scores.append(score)

    return {
        'grid_points': grid_points,
        'composition_scores': composition_scores,
        'best_composition_point': grid_points[np.argmax(composition_scores)],
        'overall_interest': np.mean(interest_map)
    }