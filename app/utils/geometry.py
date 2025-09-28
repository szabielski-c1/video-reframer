import numpy as np
import math
from typing import Tuple, List, Optional

def normalize_coordinates(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
    """Normalize pixel coordinates to 0-1 range"""
    return x / width, y / height

def denormalize_coordinates(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    """Convert normalized coordinates back to pixels"""
    return int(x * width), int(y * height)

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points"""
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """Calculate center point of bounding box [x, y, w, h]"""
    x, y, w, h = bbox
    return x + w/2, y + h/2

def calculate_bbox_area(bbox: List[float]) -> float:
    """Calculate area of bounding box"""
    _, _, w, h = bbox
    return w * h

def bbox_intersection(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate intersection area between two bounding boxes"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calculate intersection rectangle
    left = max(x1, x2)
    top = max(y1, y2)
    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)

    if left < right and top < bottom:
        return (right - left) * (bottom - top)
    return 0.0

def bbox_union(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate union area between two bounding boxes"""
    area1 = calculate_bbox_area(bbox1)
    area2 = calculate_bbox_area(bbox2)
    intersection = bbox_intersection(bbox1, bbox2)
    return area1 + area2 - intersection

def bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) for two bounding boxes"""
    intersection = bbox_intersection(bbox1, bbox2)
    union = bbox_union(bbox1, bbox2)
    return intersection / union if union > 0 else 0.0

def calculate_crop_bounds(
    center: Tuple[float, float],
    aspect_ratio: float,
    frame_width: int,
    frame_height: int,
    padding: float = 0.0
) -> Tuple[int, int, int, int]:
    """Calculate crop bounds for given center and aspect ratio"""

    center_x, center_y = center

    # Calculate crop dimensions
    if aspect_ratio > (frame_width / frame_height):
        # Crop is wider - use full height
        crop_height = frame_height
        crop_width = int(crop_height * aspect_ratio)
    else:
        # Crop is taller - use full width
        crop_width = frame_width
        crop_height = int(crop_width / aspect_ratio)

    # Apply padding
    if padding > 0:
        crop_width = int(crop_width * (1 + padding))
        crop_height = int(crop_height * (1 + padding))

    # Calculate position
    crop_x = int(center_x * frame_width - crop_width / 2)
    crop_y = int(center_y * frame_height - crop_height / 2)

    # Clamp to frame bounds
    crop_x = max(0, min(crop_x, frame_width - crop_width))
    crop_y = max(0, min(crop_y, frame_height - crop_height))

    return crop_x, crop_y, crop_width, crop_height

def constrain_point_to_bounds(
    point: Tuple[float, float],
    bounds: Tuple[float, float, float, float]
) -> Tuple[float, float]:
    """Constrain point to stay within bounds"""
    x, y = point
    min_x, min_y, max_x, max_y = bounds

    constrained_x = max(min_x, min(x, max_x))
    constrained_y = max(min_y, min(y, max_y))

    return constrained_x, constrained_y

def calculate_weighted_center(points: List[Tuple[float, float]], weights: List[float]) -> Tuple[float, float]:
    """Calculate weighted center point"""
    if not points or not weights or len(points) != len(weights):
        return 0.5, 0.5

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.5, 0.5

    weighted_x = sum(p[0] * w for p, w in zip(points, weights)) / total_weight
    weighted_y = sum(p[1] * w for p, w in zip(points, weights)) / total_weight

    return weighted_x, weighted_y

def interpolate_linear(p1: Tuple[float, float], p2: Tuple[float, float], t: float) -> Tuple[float, float]:
    """Linear interpolation between two points"""
    x1, y1 = p1
    x2, y2 = p2

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return x, y

def interpolate_bezier(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    t: float
) -> Tuple[float, float]:
    """Cubic Bezier interpolation between four control points"""

    # Bezier curve formula
    u = 1 - t
    tt = t * t
    uu = u * u
    uuu = uu * u
    ttt = tt * t

    x = (uuu * p0[0] +
         3 * uu * t * p1[0] +
         3 * u * tt * p2[0] +
         ttt * p3[0])

    y = (uuu * p0[1] +
         3 * uu * t * p1[1] +
         3 * u * tt * p2[1] +
         ttt * p3[1])

    return x, y

def calculate_bezier_control_points(
    points: List[Tuple[float, float]],
    smoothness: float = 0.3
) -> List[Tuple[float, float]]:
    """Calculate control points for smooth Bezier curve through given points"""

    if len(points) < 2:
        return points

    control_points = [points[0]]  # Start with first point

    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]

        # Calculate control points for smooth curve
        if i == 0:
            # First segment
            direction = (p1[0] - p0[0], p1[1] - p0[1])
            c1 = (p0[0] + direction[0] * smoothness, p0[1] + direction[1] * smoothness)
            c2 = (p1[0] - direction[0] * smoothness, p1[1] - direction[1] * smoothness)
        else:
            # Use previous point for direction
            prev_point = points[i - 1]
            direction = ((p1[0] - prev_point[0]) / 2, (p1[1] - prev_point[1]) / 2)
            c1 = (p0[0] + direction[0] * smoothness, p0[1] + direction[1] * smoothness)
            c2 = (p1[0] - direction[0] * smoothness, p1[1] - direction[1] * smoothness)

        control_points.extend([c1, c2, p1])

    return control_points

def calculate_trajectory_smoothness(points: List[Tuple[float, float]]) -> float:
    """Calculate smoothness metric for trajectory (lower is smoother)"""

    if len(points) < 3:
        return 0.0

    total_jerk = 0.0

    for i in range(2, len(points)):
        p0 = points[i - 2]
        p1 = points[i - 1]
        p2 = points[i]

        # Calculate second derivative (acceleration change)
        acc_x = (p2[0] - 2*p1[0] + p0[0])
        acc_y = (p2[1] - 2*p1[1] + p0[1])

        # Jerk magnitude
        jerk = math.sqrt(acc_x**2 + acc_y**2)
        total_jerk += jerk

    return total_jerk / (len(points) - 2)

def fit_aspect_ratio_crop(
    subjects: List[Tuple[float, float, float, float]],
    target_aspect: float,
    frame_width: int,
    frame_height: int,
    padding: float = 0.1
) -> Tuple[float, float]:
    """Find optimal center point to crop given subjects with target aspect ratio"""

    if not subjects:
        return 0.5, 0.5  # Default center

    # Calculate bounding box containing all subjects
    min_x = min(s[0] for s in subjects)
    min_y = min(s[1] for s in subjects)
    max_x = max(s[0] + s[2] for s in subjects)
    max_y = max(s[1] + s[3] for s in subjects)

    # Add padding
    subject_width = max_x - min_x
    subject_height = max_y - min_y

    padded_width = subject_width * (1 + padding)
    padded_height = subject_height * (1 + padding)

    # Calculate center of subjects
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Calculate required crop size for target aspect ratio
    crop_width = max(padded_width, padded_height * target_aspect)
    crop_height = max(padded_height, padded_width / target_aspect)

    # Convert to normalized coordinates
    crop_width_norm = crop_width / frame_width
    crop_height_norm = crop_height / frame_height

    # Ensure crop fits within frame
    if crop_width_norm > 1.0 or crop_height_norm > 1.0:
        # Scale down if crop is too large
        scale = min(1.0 / crop_width_norm, 1.0 / crop_height_norm)
        crop_width_norm *= scale
        crop_height_norm *= scale

    # Constrain center to keep crop within bounds
    half_crop_w = crop_width_norm / 2
    half_crop_h = crop_height_norm / 2

    center_x = max(half_crop_w, min(center_x, 1.0 - half_crop_w))
    center_y = max(half_crop_h, min(center_y, 1.0 - half_crop_h))

    return center_x, center_y

def calculate_motion_prediction(
    positions: List[Tuple[float, float]],
    timestamps: List[float],
    prediction_time: float
) -> Tuple[float, float]:
    """Predict future position based on motion history"""

    if len(positions) < 2 or len(timestamps) < 2:
        return positions[-1] if positions else (0.5, 0.5)

    # Calculate velocity from last two points
    dt = timestamps[-1] - timestamps[-2]
    if dt <= 0:
        return positions[-1]

    vx = (positions[-1][0] - positions[-2][0]) / dt
    vy = (positions[-1][1] - positions[-2][1]) / dt

    # Simple linear prediction
    pred_x = positions[-1][0] + vx * prediction_time
    pred_y = positions[-1][1] + vy * prediction_time

    # Clamp to valid range
    pred_x = max(0.0, min(1.0, pred_x))
    pred_y = max(0.0, min(1.0, pred_y))

    return pred_x, pred_y

def calculate_optimal_crop_center(
    primary_subject: Optional[Tuple[float, float, float, float]],
    secondary_subjects: List[Tuple[float, float, float, float]],
    text_regions: List[Tuple[float, float, float, float]],
    target_aspect: float,
    importance_weights: Tuple[float, float, float] = (0.7, 0.2, 0.1)
) -> Tuple[float, float]:
    """Calculate optimal crop center considering all elements"""

    primary_weight, secondary_weight, text_weight = importance_weights

    centers = []
    weights = []

    # Primary subject
    if primary_subject:
        center = calculate_bbox_center(primary_subject)
        centers.append(center)
        weights.append(primary_weight)

    # Secondary subjects
    for subject in secondary_subjects:
        center = calculate_bbox_center(subject)
        centers.append(center)
        weights.append(secondary_weight / len(secondary_subjects) if secondary_subjects else 0)

    # Important text regions
    for text in text_regions:
        center = calculate_bbox_center(text)
        centers.append(center)
        weights.append(text_weight / len(text_regions) if text_regions else 0)

    if not centers:
        return 0.5, 0.5

    # Calculate weighted center
    optimal_center = calculate_weighted_center(centers, weights)

    return optimal_center

def smooth_trajectory_with_physics(
    points: List[Tuple[float, float]],
    timestamps: List[float],
    max_velocity: float = 0.5,
    max_acceleration: float = 1.0
) -> List[Tuple[float, float]]:
    """Smooth trajectory respecting physics constraints"""

    if len(points) < 2:
        return points

    smoothed_points = [points[0]]  # Start with first point
    current_velocity = (0.0, 0.0)

    for i in range(1, len(points)):
        dt = timestamps[i] - timestamps[i-1] if i < len(timestamps) else 1/30  # Default 30fps

        target_point = points[i]
        current_point = smoothed_points[-1]

        # Calculate desired velocity
        desired_velocity = (
            (target_point[0] - current_point[0]) / dt,
            (target_point[1] - current_point[1]) / dt
        )

        # Limit acceleration
        acc_x = (desired_velocity[0] - current_velocity[0]) / dt
        acc_y = (desired_velocity[1] - current_velocity[1]) / dt

        acc_magnitude = math.sqrt(acc_x**2 + acc_y**2)
        if acc_magnitude > max_acceleration:
            scale = max_acceleration / acc_magnitude
            acc_x *= scale
            acc_y *= scale

        # Update velocity
        new_velocity = (
            current_velocity[0] + acc_x * dt,
            current_velocity[1] + acc_y * dt
        )

        # Limit velocity
        vel_magnitude = math.sqrt(new_velocity[0]**2 + new_velocity[1]**2)
        if vel_magnitude > max_velocity:
            scale = max_velocity / vel_magnitude
            new_velocity = (new_velocity[0] * scale, new_velocity[1] * scale)

        # Calculate new position
        new_point = (
            current_point[0] + new_velocity[0] * dt,
            current_point[1] + new_velocity[1] * dt
        )

        # Constrain to valid bounds
        new_point = constrain_point_to_bounds(new_point, (0.0, 0.0, 1.0, 1.0))

        smoothed_points.append(new_point)
        current_velocity = new_velocity

    return smoothed_points