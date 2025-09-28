import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TrajectorySmoothing:
    """Advanced trajectory smoothing algorithms for camera movement"""

    @staticmethod
    def gaussian_smooth(values: List[float], sigma: float = 1.0) -> List[float]:
        """Apply Gaussian smoothing to trajectory"""

        if len(values) < 3:
            return values.copy()

        # Create Gaussian kernel
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = np.arange(-kernel_size//2 + 1, kernel_size//2 + 1)
        kernel = np.exp(-x**2 / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)

        # Apply convolution with padding
        padded_values = np.pad(values, (kernel_size//2, kernel_size//2), mode='edge')
        smoothed = np.convolve(padded_values, kernel, mode='valid')

        return smoothed.tolist()

    @staticmethod
    def savitzky_golay_smooth(
        values: List[float],
        window_length: int = 5,
        polyorder: int = 3
    ) -> List[float]:
        """Apply Savitzky-Golay smoothing filter"""

        if len(values) < window_length:
            return values.copy()

        # Ensure window length is odd
        if window_length % 2 == 0:
            window_length += 1

        # Ensure polyorder is less than window length
        polyorder = min(polyorder, window_length - 1)

        try:
            smoothed = savgol_filter(values, window_length, polyorder)
            return smoothed.tolist()
        except Exception as e:
            logger.warning(f"Savitzky-Golay smoothing failed: {e}")
            return values.copy()

    @staticmethod
    def exponential_smooth(values: List[float], alpha: float = 0.3) -> List[float]:
        """Apply exponential smoothing"""

        if not values:
            return []

        smoothed = [values[0]]

        for i in range(1, len(values)):
            smoothed_value = alpha * values[i] + (1 - alpha) * smoothed[i-1]
            smoothed.append(smoothed_value)

        return smoothed

    @staticmethod
    def cubic_spline_smooth(
        values: List[float],
        timestamps: List[float],
        smoothing_factor: float = 0.1
    ) -> List[float]:
        """Apply cubic spline smoothing"""

        if len(values) != len(timestamps) or len(values) < 4:
            return values.copy()

        try:
            # Create spline with smoothing
            spline = interpolate.UnivariateSpline(
                timestamps, values,
                s=smoothing_factor * len(values)
            )

            # Evaluate spline at original timestamps
            smoothed = spline(timestamps)
            return smoothed.tolist()

        except Exception as e:
            logger.warning(f"Cubic spline smoothing failed: {e}")
            return values.copy()

    @staticmethod
    def minimum_jerk_trajectory(
        waypoints: List[Tuple[float, float]],
        timestamps: List[float],
        output_fps: float = 30.0
    ) -> Tuple[List[float], List[float], List[float]]:
        """Generate minimum jerk trajectory between waypoints"""

        if len(waypoints) != len(timestamps) or len(waypoints) < 2:
            x_coords = [p[0] for p in waypoints]
            y_coords = [p[1] for p in waypoints]
            return x_coords, y_coords, timestamps.copy()

        # Generate output timestamps
        start_time = timestamps[0]
        end_time = timestamps[-1]
        duration = end_time - start_time
        num_frames = int(duration * output_fps) + 1
        output_times = np.linspace(start_time, end_time, num_frames)

        try:
            # Separate x and y coordinates
            x_coords = [p[0] for p in waypoints]
            y_coords = [p[1] for p in waypoints]

            # Create minimum jerk splines
            x_tck = interpolate.splrep(timestamps, x_coords, s=0)
            y_tck = interpolate.splrep(timestamps, y_coords, s=0)

            # Evaluate at output timestamps
            smooth_x = interpolate.splev(output_times, x_tck)
            smooth_y = interpolate.splev(output_times, y_tck)

            return smooth_x.tolist(), smooth_y.tolist(), output_times.tolist()

        except Exception as e:
            logger.error(f"Minimum jerk trajectory generation failed: {e}")
            # Fallback to linear interpolation
            x_coords = [p[0] for p in waypoints]
            y_coords = [p[1] for p in waypoints]
            smooth_x = np.interp(output_times, timestamps, x_coords)
            smooth_y = np.interp(output_times, timestamps, y_coords)
            return smooth_x.tolist(), smooth_y.tolist(), output_times.tolist()

    @staticmethod
    def velocity_constrained_smooth(
        positions: List[Tuple[float, float]],
        timestamps: List[float],
        max_velocity: float = 0.5,
        max_acceleration: float = 1.0
    ) -> List[Tuple[float, float]]:
        """Smooth trajectory with velocity and acceleration constraints"""

        if len(positions) != len(timestamps) or len(positions) < 2:
            return positions.copy()

        smoothed = [positions[0]]
        current_velocity = np.array([0.0, 0.0])

        for i in range(1, len(positions)):
            dt = timestamps[i] - timestamps[i-1]
            if dt <= 0:
                dt = 1/30  # Default frame time

            current_pos = np.array(smoothed[-1])
            target_pos = np.array(positions[i])

            # Calculate desired velocity
            desired_velocity = (target_pos - current_pos) / dt

            # Limit acceleration
            acceleration = (desired_velocity - current_velocity) / dt
            acc_magnitude = np.linalg.norm(acceleration)

            if acc_magnitude > max_acceleration:
                acceleration = acceleration * (max_acceleration / acc_magnitude)

            # Update velocity
            new_velocity = current_velocity + acceleration * dt

            # Limit velocity
            vel_magnitude = np.linalg.norm(new_velocity)
            if vel_magnitude > max_velocity:
                new_velocity = new_velocity * (max_velocity / vel_magnitude)

            # Calculate new position
            new_position = current_pos + new_velocity * dt

            # Clamp to valid range [0, 1]
            new_position = np.clip(new_position, 0.0, 1.0)

            smoothed.append(tuple(new_position))
            current_velocity = new_velocity

        return smoothed

    @staticmethod
    def adaptive_smooth(
        values: List[float],
        motion_intensity: List[float],
        base_smoothing: float = 0.3,
        max_smoothing: float = 0.8
    ) -> List[float]:
        """Apply adaptive smoothing based on motion intensity"""

        if len(values) != len(motion_intensity) or len(values) < 3:
            return values.copy()

        smoothed = values.copy()

        for i in range(1, len(values) - 1):
            # Calculate adaptive smoothing factor
            intensity = motion_intensity[i]
            smoothing_factor = base_smoothing + (max_smoothing - base_smoothing) * (1 - intensity)

            # Apply weighted average with neighbors
            window = [values[i-1], values[i], values[i+1]]
            weights = [smoothing_factor/2, 1-smoothing_factor, smoothing_factor/2]

            smoothed[i] = sum(w * v for w, v in zip(weights, window))

        return smoothed

    @staticmethod
    def bezier_smooth(
        control_points: List[Tuple[float, float]],
        num_segments: int = 100
    ) -> List[Tuple[float, float]]:
        """Generate smooth curve using Bezier interpolation"""

        if len(control_points) < 4:
            return control_points.copy()

        def bezier_curve(p0, p1, p2, p3, t):
            """Cubic Bezier curve calculation"""
            u = 1 - t
            tt = t * t
            uu = u * u
            uuu = uu * u
            ttt = tt * t

            x = uuu * p0[0] + 3 * uu * t * p1[0] + 3 * u * tt * p2[0] + ttt * p3[0]
            y = uuu * p0[1] + 3 * uu * t * p1[1] + 3 * u * tt * p2[1] + ttt * p3[1]

            return (x, y)

        smooth_curve = []

        # Process control points in groups of 4
        for i in range(0, len(control_points) - 3, 3):
            p0 = control_points[i]
            p1 = control_points[i + 1]
            p2 = control_points[i + 2]
            p3 = control_points[i + 3]

            # Generate curve segment
            for j in range(num_segments):
                t = j / num_segments
                point = bezier_curve(p0, p1, p2, p3, t)
                smooth_curve.append(point)

        return smooth_curve

    @staticmethod
    def temporal_coherence_smooth(
        trajectories: List[List[Tuple[float, float]]],
        coherence_weight: float = 0.3
    ) -> List[List[Tuple[float, float]]]:
        """Apply temporal coherence smoothing across multiple trajectories"""

        if not trajectories or len(trajectories[0]) < 2:
            return trajectories

        smoothed_trajectories = []

        for i, trajectory in enumerate(trajectories):
            smoothed_traj = trajectory.copy()

            # Apply coherence with neighboring trajectories
            if i > 0:
                prev_traj = smoothed_trajectories[i-1]

                for j in range(len(smoothed_traj)):
                    if j < len(prev_traj):
                        # Blend with previous trajectory
                        current_point = np.array(smoothed_traj[j])
                        prev_point = np.array(prev_traj[j])

                        blended_point = (1 - coherence_weight) * current_point + coherence_weight * prev_point
                        smoothed_traj[j] = tuple(blended_point)

            smoothed_trajectories.append(smoothed_traj)

        return smoothed_trajectories

    @staticmethod
    def outlier_removal_smooth(
        values: List[float],
        threshold: float = 2.0
    ) -> List[float]:
        """Remove outliers and smooth the trajectory"""

        if len(values) < 3:
            return values.copy()

        # Calculate moving statistics
        smoothed = values.copy()
        window_size = min(5, len(values))

        for i in range(len(values)):
            # Define window around current point
            start = max(0, i - window_size // 2)
            end = min(len(values), i + window_size // 2 + 1)
            window = values[start:end]

            # Calculate local statistics
            mean_val = np.mean(window)
            std_val = np.std(window)

            # Check if current value is an outlier
            if abs(values[i] - mean_val) > threshold * std_val:
                # Replace with local median
                smoothed[i] = np.median(window)

        return smoothed

class MotionBlur:
    """Simulate and compensate for motion blur effects"""

    @staticmethod
    def estimate_motion_blur(
        velocity: Tuple[float, float],
        exposure_time: float = 1/60
    ) -> float:
        """Estimate motion blur amount based on velocity"""

        speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
        blur_amount = speed * exposure_time

        return min(blur_amount, 1.0)  # Clamp to maximum

    @staticmethod
    def compensate_for_blur(
        trajectory: List[Tuple[float, float]],
        timestamps: List[float],
        look_ahead_time: float = 0.1
    ) -> List[Tuple[float, float]]:
        """Compensate trajectory for motion blur by looking ahead"""

        if len(trajectory) != len(timestamps) or len(trajectory) < 2:
            return trajectory.copy()

        compensated = []

        for i in range(len(trajectory)):
            current_time = timestamps[i]
            target_time = current_time + look_ahead_time

            # Find future position by interpolation
            if i < len(trajectory) - 1:
                next_time = timestamps[i + 1]
                if target_time <= next_time:
                    # Interpolate between current and next
                    t = (target_time - current_time) / (next_time - current_time)
                    current_pos = np.array(trajectory[i])
                    next_pos = np.array(trajectory[i + 1])
                    future_pos = current_pos + t * (next_pos - current_pos)
                else:
                    # Use next position
                    future_pos = np.array(trajectory[i + 1])
            else:
                # Last point, no compensation
                future_pos = np.array(trajectory[i])

            compensated.append(tuple(future_pos))

        return compensated

def apply_multi_stage_smoothing(
    trajectory: List[Tuple[float, float]],
    timestamps: List[float],
    smoothing_level: float = 0.5
) -> List[Tuple[float, float]]:
    """Apply multi-stage smoothing pipeline"""

    if len(trajectory) < 3:
        return trajectory.copy()

    # Stage 1: Outlier removal
    x_coords = [p[0] for p in trajectory]
    y_coords = [p[1] for p in trajectory]

    x_clean = TrajectorySmoothing.outlier_removal_smooth(x_coords, threshold=2.0)
    y_clean = TrajectorySmoothing.outlier_removal_smooth(y_coords, threshold=2.0)

    clean_trajectory = list(zip(x_clean, y_clean))

    # Stage 2: Velocity-constrained smoothing
    max_vel = 0.3 + smoothing_level * 0.2  # Adaptive max velocity
    constrained = TrajectorySmoothing.velocity_constrained_smooth(
        clean_trajectory, timestamps, max_velocity=max_vel
    )

    # Stage 3: Final Gaussian smoothing
    x_final = [p[0] for p in constrained]
    y_final = [p[1] for p in constrained]

    sigma = smoothing_level * 2.0
    x_smooth = TrajectorySmoothing.gaussian_smooth(x_final, sigma=sigma)
    y_smooth = TrajectorySmoothing.gaussian_smooth(y_final, sigma=sigma)

    return list(zip(x_smooth, y_smooth))